import pytest
from  generator.generator import generate_response

def test_response_generation():
    context_docs = ["This is a sample FAQ about error handling."]
    query = "How to fix error codes?"
    response = generate_response(context_docs, query)
    assert len(response) > 0 
