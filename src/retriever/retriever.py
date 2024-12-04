from  embeddings.text_embeddings import text_embeddings
from  vector.vector_store import search

def retrieve_docs(file, query):
    query_embeddings = text_embeddings(query)
    print(f"Query embeddings shape: {query_embeddings.shape}")  # Debugging line
    results = search(query_embeddings)
    return results