from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain.vectorstores.qdrant import Qdrant
import uvicorn
from graph_app_sunna import react_graph,rag_agent_suna

# Define the custom embedding class
class MyEmbeddings:
    """
    Custom embedding class using SentenceTransformer.
    """
    def __init__(self, model_path="jinaai/jina-embeddings-v3"):
        self.model = SentenceTransformer(model_path, trust_remote_code=True)

    def embed_documents(self, texts):
        """
        Generate embeddings for a list of texts.
        """
        task = "retrieval.query"
        return self.model.encode(
            texts,
            task=task,
            prompt_name=task,
        )

    def __call__(self, texts):
        """
        Make the class callable for compatibility with vector stores.
        """
        return self.embed_documents(texts)


# Initialize Qdrant Client and Vector Store
client_qdrant = QdrantClient(url="http://136.243.132.228:6333")
embeddings = MyEmbeddings()
qdrant = Qdrant(
    client=client_qdrant,
    collection_name="ali_rag_final",
    embeddings=embeddings,
)

# FastAPI App Initialization
app = FastAPI()

# Request model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5  # Default number of results

# Response model
class QueryResponse(BaseModel):
    documents: List[Any]


class QueryRequestRAG(BaseModel):
    query_text: str

def query_rag_sunna(query_text):
    try:
        resp = rag_agent_suna(query=query_text)
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the API endpoint
@app.post("/query_agent_rag_sunna")
async def handle_query(request: QueryRequestRAG):
    response_text = query_rag_sunna(request.query_text)
    return {"response": response_text}




@app.post("/search", response_model=QueryResponse)
async def search_similar_documents(request: QueryRequest):
    """
    Search for similar documents using Qdrant.
    """
    try:
        # Perform similarity search
        found_docs = qdrant.similarity_search(request.query, k=request.top_k)
        # Extract relevant data from the documents
        documents = [doc.page_content for doc in found_docs]
        return QueryResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during similarity search: {e}")


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Qdrant similarity search server is running"}



# Entry point to run the server on port 9061
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=9061, reload=True)
