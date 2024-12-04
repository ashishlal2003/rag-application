import pytest
from  vector.vector_store import add_to_index, search
from  embeddings.text_embeddings import text_embeddings

def test_faiss_indexing():
    text = "Sample document for indexing."
    embedding = text_embeddings(text)
    print("Embeddings are:", embedding)
    add_to_index(embedding, text)

    query = "How to fix error codes?"
    query_embedding = text_embeddings(query)
    results = search(query_embedding)
    
    assert len(results) > 0
