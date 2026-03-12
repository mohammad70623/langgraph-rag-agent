from graph import chatbot
from langchain_core.messages import HumanMessage

print("RAG Chatbot ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("Goodbye! 👋")
        break
    
    result = chatbot.invoke({
        "messages": [HumanMessage(content=user_input)]
    })
    
    print(f"Bot: {result['messages'][-1].content}\n")