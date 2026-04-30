from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from agent.prompts import JUDGE_PERSONA

class Evaluation(BaseModel):
    is_correct: bool = Field(description="True if the AI followed all instructions in the persona and user prompt.")
    score: int = Field(description="A score from 1-5 on how well the instructions were followed.")
    reasoning: str = Field(description="A detailed explanation of why the score was given.")

# 2. Set up the Judge. This is how you set up a langchain (different from langgraph) with a structured output.
judge_llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", JUDGE_PERSONA),
    ("user", """
    ### AGENT INSTRUCTIONS (PERSONA)
    {persona}

    ### USER REQUEST
    {user_input}

    ### AGENT RESPONSE
    {agent_output}
    """)
])

# Create the chain as before
judge_chain = judge_prompt | judge_llm.with_structured_output(Evaluation)
