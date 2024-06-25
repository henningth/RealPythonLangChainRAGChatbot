from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

REVIEWS_CHROMA_PATH = "chroma_data/"

reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=OllamaEmbeddings(),
)

question = """Has anyone complained about
           communication with the hospital staff?"""
relevant_docs = reviews_vector_db.similarity_search(question, k=3)

relevant_docs[0].page_content
