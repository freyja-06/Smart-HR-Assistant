from backend.graphs.main_graph import run
from backend.state.graph_state import GraphState
from dotenv import load_dotenv
load_dotenv()


state: GraphState = {}
config = {"configurable": {"thread_id": "1"}}
if __name__ == "__main__":

    while True:
        
        try:
            if not state:
                state = {"user_input": str(input("Chào HR! Bạn có yêu cầu gì? \n"))}

            else:
                user_input = str(input("Bạn còn yêu cầu gì khác không? \n"))
                state = {"user_input": user_input, "history_cv_store": state.get("history_cv_store")}
            
                result = run(state, config)
                print(result["final_answer"])
                # Nên lưu các biến nhiệm vụ để trả về kết quả cho người dùng
        
        except Exception as e:
            print(f"\n Lỗi: {e}")
            continue

        


