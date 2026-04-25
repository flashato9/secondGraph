import pytest

from agent import graph, ContextSchema,LLMConfiguration
from langchain_core.messages import HumanMessage
from agent import judge_chain

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_first_message() -> None:
    initial_state = {
        "messages": [
            HumanMessage("hello")
            ]
    }
    context_instance = ContextSchema(
        llm_configuration = LLMConfiguration(),
        persona = """
            You are a joke teller. Tell me a joke.
        """
    )
    res = await graph.ainvoke(initial_state, context = context_instance)
    assert "messages" in res
    assert res["messages"][-1].type == "ai"

    evaluation = await judge_chain.ainvoke({
        "persona": context_instance.persona,
        "output": res["messages"][-1].content
    })

    print(f"\nJudge Score: {evaluation.score}")
    print(f"Judge Reason: {evaluation.explanation}")
    
    assert evaluation.score == 1, f"Judge failed the response: {evaluation.explanation}"