from langchain_community.chat_models.ollama import ChatOllama

chat_model = ChatOllama(model="mistral", temperature=0)

from langchain.schema.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(
        content="""You're an assistant knowledgeable about
        healthcare. Only answer healthcare-related questions."""
    ),
    HumanMessage(content="What is Medicaid managed care?"),
]

response = chat_model.invoke(messages)

print(response.content)