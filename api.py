from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qna import agent

class Input(BaseModel):
    text: str
    k:int

class Output(BaseModel):
    text: str

def get_response(query,k):
    thread_config = {"configurable": {"thread_id": "some_id"}}
    result = agent.invoke({
                "query": query,
                "k":k,
                "pdf_path": "", 
                "result": "",
                "imgs": [],
                "paper_url": None,
                "next_node": None,
                "chat_history":[]
     }, config=thread_config)
    state = agent.get_state(thread_config)
    return result['result']

app = FastAPI()


@app.post("/api/v1/ask", response_model=Output)
async def endpoint(item: Input):
    try:
        response = get_response(item.text.strip(),item.k)
        return Output(text=response)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred in the module."
        )
