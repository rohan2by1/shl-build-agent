from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- ADD THIS
from contextlib import asynccontextmanager
from app.schemas import ChatRequest, ChatResponse
from app.retriever import CatalogRetriever
from app.agent import SHLAgent

# Global application state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the catalog and initialize the agent once
    print("Initializing Vector DB and Agent...")
    retriever = CatalogRetriever()
    agent = SHLAgent(retriever=retriever)
    
    app_state["agent"] = agent
    print("Application ready.")
    yield
    # Shutdown logic can go here if needed
    app_state.clear()

app = FastAPI(title="SHL Agentic Recommender", lifespan=lifespan)
# --- ADD THIS CORS BLOCK ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (good for local testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    agent: SHLAgent = app_state.get("agent")
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized.")
    
    try:
        response = agent.process_chat(request.messages)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)