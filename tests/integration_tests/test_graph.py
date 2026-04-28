import asyncio

from langgraph.types import Command
import pytest

from agent import graph, ContextSchema,LLMConfiguration
from langchain_core.messages import AIMessage, HumanMessage
from agent import judge_chain, AGENT_PERSONA

pytestmark = pytest.mark.asyncio(loop_scope="module")


# @pytest.mark.langsmith
# async def test_hello() -> None:
#     # --- ARRANGE ---
#     agent_persona = (
#             AGENT_PERSONA
#             )
#     user_prompt = "hello"
#     initial_state = {
#         "messages": [
#             HumanMessage(user_prompt)
#             ]
#     }

#     context_instance = ContextSchema(
#         llm_configuration = LLMConfiguration(),
#         persona = agent_persona
#     )
#     # --- ACT ---
#     config = {"configurable": {"thread_id": "flow_test"}}
#     res = await graph.ainvoke(initial_state,config, context = context_instance)
#     # --- ASSERT ---
#     evaluation = await judge_chain.ainvoke({
#         "persona": context_instance.persona,
#         "user_input": user_prompt,
#         "agent_output": res["messages"][-1].content
#     })
#     print(f"Agent Persona:{agent_persona}")
#     print(f"Agent Initial State: {initial_state}")
#     print(f"Agent Output: {res["messages"][-1].content}")
#     print(f"Judge Score: {evaluation.score}")
#     print(f"Judge Reasoning: {evaluation.reasoning}")
    
#     assert evaluation.is_correct, f"Judge failed the response: {evaluation.reasoning}"

# @pytest.mark.langsmith
# async def test_what_is_your_purpose() -> None:
#     # --- ARRANGE ---
#     agent_persona = (
#             AGENT_PERSONA
#             )
#     user_prompt = "What is your purpose?"
#     initial_state = {
#         "messages": [
#             HumanMessage(user_prompt)
#             ]
#     }

#     context_instance = ContextSchema(
#         llm_configuration = LLMConfiguration(),
#         persona = agent_persona
#     )
#     # --- ACT ---
#     config = {"configurable": {"thread_id": "flow_test"}}
#     res = await graph.ainvoke(initial_state,config, context = context_instance)
#     # --- ASSERT ---
#     evaluation = await judge_chain.ainvoke({
#         "persona": context_instance.persona,
#         "user_input": user_prompt,
#         "agent_output": res["messages"][-1].content
#     })
#     print(f"Agent Persona:{agent_persona}")
#     print(f"Agent Initial State: {initial_state}")
#     print(f"Agent Output: {res["messages"][-1].content}")
#     print(f"Judge Score: {evaluation.score}")
#     print(f"Judge Reasoning: {evaluation.reasoning}")
    
#     assert evaluation.is_correct, f"Judge failed the response: {evaluation.reasoning}"

# @pytest.mark.langsmith
# async def test_how_do_you_work() -> None:
#     # --- ARRANGE ---
#     agent_persona = (
#             AGENT_PERSONA
#             )
#     user_prompt = "How do you work?"
#     initial_state = {
#         "messages": [
#             HumanMessage(user_prompt)
#             ]
#     }

#     context_instance = ContextSchema(
#         llm_configuration = LLMConfiguration(),
#         persona = agent_persona
#     )
#     # --- ACT ---
#     config = {"configurable": {"thread_id": "flow_test"}}
#     res = await graph.ainvoke(initial_state,config, context = context_instance)
#     # --- ASSERT ---
#     evaluation = await judge_chain.ainvoke({
#         "persona": context_instance.persona,
#         "user_input": user_prompt,
#         "agent_output": res["messages"][-1].content
#     })
#     print(f"Agent Persona:{agent_persona}")
#     print(f"Agent Initial State: {initial_state}")
#     print(f"Agent Output: {res["messages"][-1].content}")
#     print(f"Judge Score: {evaluation.score}")
#     print(f"Judge Reasoning: {evaluation.reasoning}")
    
#     assert evaluation.is_correct, f"Judge failed the response: {evaluation.reasoning}"

@pytest.mark.langsmith
async def test_conversation() -> None:
    # --- ARRANGE ---
    agent_persona = (
            AGENT_PERSONA
            )
    context_instance = ContextSchema(
        llm_configuration = LLMConfiguration(),
        persona = agent_persona
    )
    config = {"configurable": {"thread_id": "flow_test"}}

    user_messages = [
        # { "messages": [HumanMessage("hello")]  },
        # { "messages": [HumanMessage("What is your purpose?")] },
        # { "messages": [HumanMessage("how do I use you?")] },
        # { "messages": [HumanMessage("how do I provide you a picture?")] },
        { "messages": [HumanMessage("Here is my room: 'C:\\Users\\Ato_K\\Downloads\\living_room.jpg'")] }
    ]
    num = 0
    for message in user_messages:
        num += 1
        print(f"==================================== Round {num} ===================================")
        # --- ACT ---
        res = await graph.ainvoke(message,config, context = context_instance)
        state = await graph.aget_state(config)
        while state.next :
            res = await graph.ainvoke(
                Command(resume="Approve"), 
                config=config, 
                context=context_instance
            )
            # Refresh state to see if there are more interrupts
            state = await graph.aget_state(config)
        # --- PRINT ---
        for result_message in res["messages"]:
            result_message.pretty_print()        
        # --- ASSERT ---
        evaluation = await judge_chain.ainvoke({
            "persona": context_instance.persona,
            "user_input": message,
            "agent_output": res["messages"][-1].content
        })
        print("================================== Judge Score ==================================")
        print(f"{evaluation.score}")
        print("================================== Judge Reasoning ==================================")
        print(f"{evaluation.reasoning}")
        assert evaluation.is_correct, f"Judge failed the test."