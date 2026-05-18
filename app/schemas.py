from pydantic import BaseModel, Field
from typing import List, Literal

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str = Field(..., description="The title of the assessment")
    url: str = Field(..., description="The URL to the assessment")
    test_type: str = Field(..., description="The test type code (e.g., K, P, A)")

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = []
    end_of_conversation: bool