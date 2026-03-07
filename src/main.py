"""
Luồng hoạt động của chương trình:

    1. Người dùng (doanh nghiệp) nạp dữ liệu cv và policies, thực hiện nạp dữ liệu và lưu vào vector database
    2. Người dùng thực hiện truy vấn
    3. Agent nhận yêu cầu truy vấn, tự quyết định việc truy vấn (sử dụng rag) hay dừng những tool khác
        Trường hợp truy vấn không cần sử dụng tool, agent trả về kết quả cho người dùng
        Trường hợp truy vấn cần sử dụng tool, gửi trả lại kết quả về cho agent.
    4. Chờ người dùng thực hiện truy vấn tiếp theo, quay lại từ bước 2


"""
from agents.agent_chain import chain
from agents.tools import tools
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
# Sau này sẽ sử dụng Dùng SqliteSaver hoặc DB tương tự

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def agent_node(state: AgentState):
    print("\n-> [Agent] Đang suy nghĩ...")

    response = chain.invoke({
        "messages": state["messages"]
    })

    return {"messages": [response]}

tools_node = ToolNode(tools)


# Luồng chạy graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# Biên dịch và chạy dồ thị

app = workflow.compile(checkpointer= memory)

user_input = {
    "messages": [HumanMessage(content=str(input("Nhập yêu cầu: ")))]
}

config = {"configurable": {"thread_id": "1"}} 
result = app.invoke(user_input, config=config)

print(result["messages"][-1].content)