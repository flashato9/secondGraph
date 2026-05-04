from langchain_core.runnables import Runnable
from langchain_core.language_models import  LanguageModelInput

from langchain_core.messages import AIMessage

LLM = Runnable[LanguageModelInput, AIMessage]