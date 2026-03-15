from langchain_core.prompts import ChatPromptTemplate
from llm_processor.llm import llm

optimize_query_instruction = """


"""

context_building_instruction = """


"""


optimize_query_prompt = ChatPromptTemplate.from_messages([
    ("system", f"{optimize_query_instruction}"),
    ("placeholder", "{messages}")
])
optimize_query_chain = optimize_query_prompt | llm


context_building_prompt = ChatPromptTemplate.from_messages([
    ("system", f"{context_building_instruction}"),
    ("placeholder", "{messages}")
])
context_building_chain = context_building_prompt | llm


