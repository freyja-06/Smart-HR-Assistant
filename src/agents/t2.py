from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3",
    temperature=0.7
)

question = "Explain machine learning in simple terms"

response = llm.invoke(question)

print(response.content)