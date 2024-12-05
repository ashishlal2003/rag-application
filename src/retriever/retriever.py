from  embeddings.text_embeddings import text_embeddings
from  vector.vector_store import search

def retrieve_docs(file, query):
    query_embeddings = text_embeddings(query)
    results = search(query_embeddings)
    return results