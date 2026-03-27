from backend.graphs.main_graph import run
import asyncio
from backend.state.graph_state import GraphState
from dotenv import load_dotenv
load_dotenv()


async def main():
    state: GraphState = {}
    config = {"configurable": {"thread_id": "1"}}
    while True:
        if not state:
            state = {"user_input": str(input("Chào HR! Bạn có yêu cầu gì? \n"))}
        else:
            user_input = str(input("Bạn còn yêu cầu gì khác không? \n"))
            if user_input.lower() == "không":
                break
            state = {"user_input": user_input, "history_cv_store": state.get("history_cv_store")}
        
        result = await run(state, config)
        print(result["final_answer"])

if __name__ == "__main__":
    asyncio.run(main())