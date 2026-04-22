# main.py - Entry point for AutoStream agent

from src.agent import build_graph, AgentState
from langchain_core.messages import HumanMessage

def run_agent():
    print("\n" + "="*50)
    print("  Welcome to AutoStream AI Assistant!")
    print("  Type 'quit' or 'exit' to end the chat.")
    print("="*50 + "\n")

    graph = build_graph()

    # Initial state
    state: AgentState = {
        "messages": [],
        "intent": "",
        "collecting_lead": False,
        "lead_step": "name"
    }

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("\nThanks for chatting with AutoStream. Goodbye!")
            break

        # Add user message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run through the graph
        state = graph.invoke(state)

        # Print last AI message
        last_message = state["messages"][-1].content
        print(f"\nAutoStream: {last_message}\n")

if __name__ == "__main__":
    run_agent()