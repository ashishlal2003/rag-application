from vector.vector_store import add_to_index
from embeddings.text_embeddings import text_embeddings

def ingest_text(file_path):
    with open(file_path, "r") as file:
        documents = file.readlines()
        for doc in documents:
            embedding = text_embeddings(doc)
            add_to_index(embedding, doc)  
    return documents