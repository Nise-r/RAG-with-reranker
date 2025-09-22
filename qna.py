from rapidocr import RapidOCR
from PIL import Image
import io
import json
from groq import Groq
import base64
import time
import pymupdf
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
from langchain.schema import BaseMessage
from langchain.chat_models import init_chat_model
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.documents import Document

from typing import TypedDict,List
from pydantic import BaseModel, Field

import re
import tabula
from collections import Counter

from langgraph.types import Command, interrupt
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from werkzeug.utils import secure_filename
from rank_bm25 import BM25Okapi
import math

    
log_messages = []
ret_chunks = []
api_key = "YOUR_GROQ_API_KEY"

class Parser:
    def __init__(self,api_key:str):
        self._api_key = api_key 
        self._engine = RapidOCR()
        with open('./sources.json','r') as f:
            self._sources = json.load(f)
            
    def _extract_captions_of_images(self,doc,page):
        imgs = page.get_images()
        client = Groq(api_key=self._api_key)
        captions = {}

        for i in range(len(imgs)):
            print("...VLM Called...",end="")

            xref = imgs[i][0]
            base_image = doc.extract_image(xref)

            #if image is unicolor that means it is either mask or artifact
            if base_image['colorspace']==1:
                continue

            image_bytes = base_image["image"]

            image_ext = base_image["ext"]

            image = Image.open(io.BytesIO(image_bytes))
            image = image.resize((360,180))
            output = io.BytesIO()
            # image
            image.save(output, format=image_ext)
            base64_image = base64.b64encode(output.getvalue()).decode('utf-8')

            start_time = time.time()
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image in no more than 100 words as much as possible/"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
            )

            captions[f"img_{xref}"] = chat_completion.choices[0].message.content

        return captions
        
    def _extract_using_ocr(self,page):
        print("...OCR called...",end="")
        img = page.get_pixmap()
        image_bytes = img.tobytes()
        image = Image.open(io.BytesIO(image_bytes))

        if image.width > image.height:
            image = image.rotate(90,expand=True)

        image = image.resize((400,800))
        result = self._engine(image)
        text = "\n".join(txt for txt in result.txts)
        
        return text

    def _extract_text_excluding_tables(self,page):
        tables = page.find_tables(strategy="lines_strict")
        table_bboxes = [table.bbox for table in tables]

        def is_inside_any_table_bbox(bbox):
            for table_bbox in table_bboxes:
                # print(table_bbox)
                if pymupdf.Rect(table_bbox).intersects(pymupdf.Rect(bbox)):
                    return True
            return False

        blocks = page.get_text("blocks")  
        filtered_text = [
            block[4] for block in blocks
            if not is_inside_any_table_bbox(block[:4])
        ]

        return "\n".join(filtered_text)
    
    def _extract_table_content(self,page):
        tables = page.find_tables()
        tables_list = [table.to_markdown() for table in tables]

        text = "\n".join(text for text in tables_list)

        return text
    def _get_table_from_pg(self,pdf_path,pg):
        tables = tabula.read_pdf(pdf_path,pages=str(pg+1),multiple_tables=True)
        return tables
    
    def _extract_formulas_from_text(self,text):
        formulas = []

        # 1. LaTeX inline math: $...$
        inline_latex = re.findall(r'\$(.+?)\$', text)
        formulas.extend([f.strip() for f in inline_latex])

        # 2. LaTeX display math: \[...\]
        display_latex = re.findall(r'\\\[(.+?)\\\]', text, flags=re.DOTALL)
        formulas.extend([f.strip() for f in display_latex])

        # 3. LaTeX equation environments
        env_latex = re.findall(r'\\begin{equation\*?}(.+?)\\end{equation\*?}', text, flags=re.DOTALL)
        formulas.extend([f.strip() for f in env_latex])

        # 4. LaTeX align environments
        align_envs = re.findall(r'\\begin{align\*?}(.+?)\\end{align\*?}', text, flags=re.DOTALL)
        formulas.extend([f.strip() for f in align_envs])

        # 5. ASCII/Unicode math heuristics (e.g., x^2 + y^2 = z^2 or x² + y² = z²)
        # Look for lines with multiple math symbols or variables
        math_lines = []
        for line in text.splitlines():
            if re.search(r'[a-zA-Z0-9][\^²³√±*/=<>+\-]+[a-zA-Z0-9]', line):
                if len(line.strip()) > 5:  # avoid noise
                    math_lines.append(line.strip())

        # Filter duplicates and obvious non-formulas
        for line in math_lines:
            if line not in formulas and not line.startswith('Figure') and '=' in line:
                formulas.append(line)

        return formulas
    
    
    def _common_font_size(self,pdf_path):
        doc = pymupdf.open(pdf_path)
        font_sizes = []

        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for line in b["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span["size"])
        counter = Counter(font_sizes)
        return counter.most_common()[0][0]

    def _format_headings(self,headings):
        prev_y = 0
        result = ""
        for heading in headings:
            if heading['bbox'][1]!=prev_y:
                result += "\n"
            result+=heading['text']+" "
            prev_y = heading['bbox'][1]
        return result

    def _get_headings(self,page,comm_font_size):
        headings = []
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_size = round(span.get("size", 0))
                    font_flags = span.get("flags", 0)
                    text = span.get("text", "").strip()

                        # Skip empty strings
                    if not text:
                        continue

                        # Heuristic: large font size is probably a heading
                    if font_size > round(comm_font_size) or (font_size == round(comm_font_size) and (font_flags & pymupdf.TEXT_FONT_BOLD or "Bold" in span.get("font", ""))):
                        headings.append({
                            "text": text,
                            "size": font_size,
                            "font": span.get("font"),
                            "flags": font_flags,
                            "bbox": span.get("bbox"),
                        })

        return self._format_headings(headings)


    def parse_pdf(self,path):
        global log_messages
        log_messages.append("Parsing the pdf")
        doc = pymupdf.open(path)
        parsed = []
        comm_font_size = self._common_font_size(path)
        
        title=''
        source=''
        check = path.split('/')[-1]
        for d in self._sources:
            if d['path'].split('/')[-1]==check:
                title = d['title']
                source = d['url']
                break

        for i in range(doc.page_count):
            print(f"Page {i+1}",end="")

            full_pg = {}
            start_time = time.time()
            pg = doc.load_page(i)

            text = self._extract_text_excluding_tables(pg)
            
            if text == "" or text == []:
                text = self._extract_using_ocr(pg)
                img = {}
                table = ""
            else:
#                 img = self._extract_captions_of_images(doc,pg)
                img = {}
                table = self._extract_table_content(pg)
#                 table = self._get_table_from_pg(path,i)
                headings = self._get_headings(pg,comm_font_size)

            full_pg['text'] = text
            full_pg['tables'] = table
            full_pg['imgs'] = img
            full_pg['title'] = title
            full_pg['source'] = source
            full_pg['page'] = i+1
            full_pg['headings'] = headings
            full_pg['formulas'] = self._extract_formulas_from_text(text)
        
            parsed.append(full_pg)
            print(f"..Done.. {time.time()-start_time}")

        log_messages.append("PDF parsed")
        return parsed
    
    

class AgentState(TypedDict):
    query: str
    k:int
    pdf_path: list[str]
    result: str
    imgs: list[str]
    paper_url: str
    next_node: str
    prev_node: str
    chat_history: List[BaseMessage]         

        
def rerank(topk,query):
    alpha = 0.6
    
    corpus_for_bm = [result[0].page_content.split(" ") for result in topk]

    bm25 = BM25Okapi(corpus_for_bm)
    doc_scores = bm25.get_scores(query.split(" "))
    doc_score_nm = [(2 / math.pi) * math.atan(score) for score in doc_scores]
    
    reranked = []
    for i,top in enumerate(topk):
        final_score = (alpha*topk[i][1])+((1-alpha)*doc_score_nm[i])
        
        reranked.append([topk[i][0],topk[i][1],doc_score_nm[i],final_score])
        
    reranked = sorted(reranked,key = lambda x:x[3],reverse=True)
    return reranked

class Agent:
    def __init__(self,vdb_name:str,vdb_path:str,api_key:str):
        self._api_key = api_key
        self._embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

        self._vectorstore = Chroma(
            collection_name=vdb_name,
            embedding_function=self._embedding,
            persist_directory=vdb_path,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        
        self._parser = Parser(api_key=api_key)
        
        self._text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=250
        )
        
        self._llm = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct",temperature=0.2,api_key= api_key)
    

    def _parse_and_embed(self,state:AgentState):
        global log_messages
#         log_messages.append("In parse and embed")
        
        print("In parse_and_embed")
        
        for pdf in state['pdf_path']:
            print(f"Parsing {pdf}")
            result = self._parser.parse_pdf(pdf)
            pdf_path = "/".join(pdf.split('/')[:-1])+"/"+secure_filename(pdf.split('/')[-1])
        
            title,source = result[0]['title'],result[0]['source']
            
            docs_list = [Document(page_content=page['text']+"\n\n"+page['tables']
                      +"\n\n"+"\n".join(page['imgs'][key] for key in page['imgs'].keys())+"\n\n"+
                      page['headings']+"\n\n"+'\n'.join(formula for formula in page['formulas']),
                          metadata={"page": page['page'],"imgs":False if not page['imgs'] else ",".join(img.split('_')[1] for img in page['imgs']), 
                                   'pdf_path':pdf,"title":title,"source":source,"headings":','.join(heading for heading in page['headings'].split('\n'))}) for page in result]






            doc_splits = self._text_splitter.split_documents(docs_list)

            self._vectorstore.add_documents(documents=doc_splits)
        
            print(f"parsed and embedded the pdf {pdf}")

            return {'next_node':"rag_and_generate",'prev_node':'parse_and_embed'}

   
    def _rag_and_generate(self,state:AgentState):
        global log_messages, ret_chunks
        print("In rag_and_generate")
       
        docs = self._vectorstore.similarity_search_with_relevance_scores(state['query'],k=state['k']*2)
#         print(docs)
        reranked = rerank(docs,state['query'])

        content = "\n\n".join(docs[0].page_content+"\n\n\n"+"PDF Title:"+docs[0].metadata['title']+'\nPDF Source:'+docs[0].metadata['source']+"\nPage:"+str(docs[0].metadata['page']) for docs in reranked[:state['k']])
        
        for t in reranked[:state['k']]:
            z = t[0].to_json()
            z['similarity_score'] = t[1]
            ret_chunks.append(z)
        
        if reranked[0][3]>0.4:
            prompt = ChatPromptTemplate.from_messages([
                    ("system","""Use the following pieces of context to answer the question at the end.
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    Keep the answer as concise as possible. Try to explain as descriptive as possible.
                    Stay true to the contents. Make sure the answer is long and descriptive.
                    Always say "thanks for asking!" at the end of the answer
                    Cite the Title and Source Url at the bottom of response  from all the context."""),
                    MessagesPlaceholder('chat_history'),
                    (
                        "human", [
                            {"type": "text", "text": "Context: {context}\n\nInput: {input}"},
                        ]
                    )
            ])
            chain = prompt | self._llm

            result = chain.invoke({
                "input":state['query'],
                "chat_history":state['chat_history'],
                "context":content
            })
            state["chat_history"].append(HumanMessage(content=state['query']))
            state["chat_history"].append(AIMessage(content=result.content))

            return {'result': result.content,'imgs':[],'prev_node':"rag_and_generate"} 
            
        
        else:
            print('='*20)
            print("Score: ",reranked[0][3])
            print('='*20)
            
            return {'result': "Seems, like don't have much information to answer that question.",'imgs':[],'prev_node':"rag_and_generate"}  
            
        
            
        
         

    def _router(self,state:AgentState):
        print("In router")
        
        if state['pdf_path'] == None or state['pdf_path']==[]:
            return {'next_node':'rag_and_generate','prev_node':'router'}
        else:
            return {'next_node':'parse_and_embed','prev_node':'router'}
    

    def create_agent(self):
        graph = StateGraph(AgentState)
        checkpointer = MemorySaver()

        graph.add_node("Router",self._router)
        graph.add_node("parse_and_embed",self._parse_and_embed)
        graph.add_node("rag_and_generate",self._rag_and_generate)
      
        
        graph.set_entry_point("Router")


        graph.add_edge("parse_and_embed","rag_and_generate")
        graph.add_edge("rag_and_generate",END)

        graph.add_conditional_edges(
            "Router",
            lambda state: state['next_node'],
            {
                "parse_and_embed":"parse_and_embed",
                "rag_and_generate":"rag_and_generate",
            }
        )

        return graph.compile(checkpointer=checkpointer)


agent=Agent(vdb_name="industrial_docs2",vdb_path="./industrial_docs2",api_key=api_key)

agent = agent.create_agent()
