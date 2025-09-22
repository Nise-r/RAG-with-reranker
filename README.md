# RAG With Reranker

## 📌Overview

This project implements the logic of hybrid reranker that blends the "vector score" and "keyword score" to get a normalized final score. On which the retrieved documents from Vector Database are then sorted.

This project show the comparison between normal RAG system and RAG system with reranker.

## 🚀 Features

- 📎 Upload and parse multiple PDF documents 
- 💬 Ask natural language questions based on the document
- 🧠 Powered by a LangGraph-based intelligent agent
- ⚡ Parses text, tables, headings, formulas using PyMuPdf and captions Images using the VLM.
- 🤖 Uses multimodal LLM(meta-llama/llama-4-scout-17b-16e-instruct).
- 🔍 **RAG pipeline** with:
  - 🧠 ChromaDB vector store 
  - 🎯 Hybrid reranking used Cosine similarity + BM25 Score for reranking.
- 🌐 Simple and responsive web UI (Flask + Vanilla JS)
      
## 🔄Agent Workflow:

<img width="236" height="432" alt="Unknown" src="https://github.com/user-attachments/assets/929b20dd-9906-4efd-90bb-71e616bf4b20" />
    
## 📂Project Structure:
```bash
├── industrial_docs2/                    # Choma DB with some documents embedded
├── industrial-safety-pdfs/              # PDFs
├── static/                              # Contains css file
├── templates/                           # Contains html file
├── uploads/                             # Contains PDFs uploaded through Flask Web UI
├── README.md                            # Documentation
├── requirements.txt                     # Dependencies
├── 8questions.txt                       # txt file containing 8 questions for comparison
├── comparison.pdf                       # PDF file containing table of comparision between ranked and non-ranked RAG system.
├── mini-rag+reranker.ipynb              # Jupyter notebook
├── qna_without_reranker.py              # Agent without reranker
├── qna.py                               # Agent with reranker
├── sources.json                         # Sources for PDFs
├── app.py                               # Flask file for Web UI
└── api.py                               # Fastapi file for api endpoint
```

## 📊 Results:
- comparison.pdf has table that shows the difference betweeen respnoses of RAG system with and without Reranker.
- Hybrid Reranker approach improves the response of the RAG system.
- 8questions.txt has list of 8 questions on which the comparison is made.

## 🛠️ Reranker Code:
```bash
# This function takes list of topk docs retrieved and calculate the hybrid score(cosine similarity + BM25 score) then rank them.
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
```
## 🚀Installation & Setup:
```bash
#Clone repository
git clone https://github.com/Nise-r/RAG-with-reranker.git
cd RAG-with-reranker

# Install dependencies
pip install -r requirements.txt

# Set API key in file directly
# qna.py & qna_without_reranker.py line 36
api_key = "YOUR_GROQ_API_KEY"

#run the flask app
python app.py

```

## ▶️Usage Example:
```bash
#For Flask Web application:
python app.py

#For fastapi server:
fastapi dev api.py

#After setting up fastapi server you can send request to server:
curl -X POST http://127.0.0.1:8000/api/v1/ask -H "Content-Type: application/json" -d '{"text":"YOUR_QUERY","k":"INT_REPRESENTING_DOCS_RETRIEVED"}'
```
## ▶️Example Curl Requests:
```bash
#After setting up fastapi server you can send request to server:
curl -X POST http://127.0.0.1:8000/api/v1/ask -H "Content-Type: application/json" -d '{"text":"what are the Methods of Safegaurding?","k":"4"}'


curl -X POST http://127.0.0.1:8000/api/v1/ask -H "Content-Type: application/json" -d '{"text":"Give Steps to meet Machinery Directive requirements","k":"4"}'
```
