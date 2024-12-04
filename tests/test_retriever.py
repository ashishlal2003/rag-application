import pytest
from  retriever.retriever import retrieve_docs

def test_retrieval():
    file_path = "data/text/sample.txt"
    query = "How to fix error codes?"
    retrieved_docs = retrieve_docs(file_path, query)
    assert len(retrieved_docs) > 0 
