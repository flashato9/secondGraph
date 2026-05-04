import asyncio
from email.mime import message
from logging import config

from langgraph.types import Command
import pytest

from langgraph.checkpoint.memory import MemorySaver
from agent import ContextSchema,LLMConfiguration,get_graph
from langchain_core.messages import AIMessage, HumanMessage
from agent import judge_chain, AGENT_PERSONA

pytestmark = pytest.mark.asyncio(loop_scope="module")

memory = MemorySaver()
graph = get_graph().compile(checkpointer= memory)

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

# @pytest.mark.langsmith
# async def test_conversation() -> None:
#     # --- ARRANGE ---
#     agent_persona = (
#             AGENT_PERSONA
#             )
#     context_instance = ContextSchema(
#         llm_configuration = LLMConfiguration(),
#         persona = agent_persona
#     )
#     config = {"configurable": {"thread_id": "flow_test"}}

#     user_messages = [
#         { "messages": [HumanMessage("hello")]  },
#         { "messages": [HumanMessage("What is your purpose?")] },
#         { "messages": [HumanMessage("how do I use you?")] },
#         { "messages": [HumanMessage("how do I provide you a picture?")] },
#         { "messages": [HumanMessage("Here is my room: 'static/dirty_room.jpg'")] }
#     ]
#     num = 0
#     for message in user_messages:
#         num += 1
#         print(f"==================================== Round {num} ===================================")
#         # --- ACT ---
#         res = await graph.ainvoke(message,config, context = context_instance)
#         state = await graph.aget_state(config)
#         while state.next :
#             res = await graph.ainvoke(
#                 Command(resume="Approve"), 
#                 config=config, 
#                 context=context_instance
#             )
#             # Refresh state to see if there are more interrupts
#             state = await graph.aget_state(config)
#         # --- PRINT ---
#         for result_message in res["messages"]:
#             result_message.pretty_print()        
#         # --- ASSERT ---
#         evaluation = await judge_chain.ainvoke({
#             "persona": context_instance.persona,
#             "user_input": message,
#             "agent_output": res["messages"][-1].content
#         })
#         print("================================== Judge Score ==================================")
#         print(f"{evaluation.score}")
#         print("================================== Judge Reasoning ==================================")
#         print(f"{evaluation.reasoning}")
#         assert evaluation.is_correct, f"Judge failed the test."

# def encode_image(image_path):
#     import base64
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')
    
# async def test_image_updload():
#     # Create the content list
#     image_base64 = encode_image("static/dirty_room.jpg")
#     # --- ARRAGE ---
#     agent_persona = (
#             AGENT_PERSONA
#             )
#     context_instance = ContextSchema(
#         llm_configuration = LLMConfiguration(),
#         persona = agent_persona
#     )
#     config = {"configurable": {"thread_id": "flow_test"}}

#     user_messages ={
#         "messages":[
#             HumanMessage(
#                     content = [
#                         {
#                             "type": "text", 
#                             "text": "Describe the cleliness of this room."
#                         },
#                         {
#                             "type": "image_url",
#                             "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
#                         }
#                     ]
#                 ),
#         ]
#     }
#     # --- ACT ---
#     res = await graph.ainvoke(user_messages,config, context = context_instance)
#     # --- PRINT ---
#     for result_message in res["messages"]:
#         result_message.pretty_print()     
#     # --- ASSERT ---
#     evaluation = await judge_chain.ainvoke({
#             "persona": context_instance.persona,
#             "user_input": user_messages,
#             "agent_output": res["messages"][-1].content
#         })
#     print("================================== Judge Score ==================================")
#     print(f"{evaluation.score}")
#     print("================================== Judge Reasoning ==================================")
#     print(f"{evaluation.reasoning}")
#     assert evaluation.is_correct, f"Judge failed the test."