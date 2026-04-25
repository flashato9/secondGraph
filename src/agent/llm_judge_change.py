from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 1. Define what the Judge is looking for
class Grade(BaseModel):
    score: int = Field(description="1 if it's a joke, 0 if not")
    explanation: str = Field(description="Explain why you gave this score")

# 2. Set up the Judge
judge_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        ).with_structured_output(Grade)

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an impartial judge. Your job is to verify if the AI told a joke based on the user's persona."),
    ("user", "Persona: {persona}\n\nAI Response: {output}")
])

judge_chain = judge_prompt | judge_llm