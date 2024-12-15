from graph_app_sunna import react_graph,rag_agent
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph_app_sunna import react_graph

# Define a request model for incoming queries
class QueryRequest(BaseModel):
    query_text: str

# Initialize the FastAPI app
app = FastAPI()

# Define the RAG query function
def query_rag(query_text):
    try:
        resp = rag_agent(query=query_text)
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the API endpoint
@app.post("/query_agent")
async def handle_query(request: QueryRequest):
    response_text = query_rag(request.query_text)
    return {"response": response_text}