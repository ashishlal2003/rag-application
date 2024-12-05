import faiss
import numpy as np

# FAISS Index initialization (ensure the dimension is correct)
index = faiss.IndexFlatIP(384)  # Using 384 for example, change according to your model
documents = []  # List to store documents

def add_to_index(embedding, doc):
    """
    Add an embedding and its corresponding document to the FAISS index.
    """
    global documents
    embedding = np.array([embedding]).astype('float32')
    
    index.add(embedding)  
    documents.append(doc) 

def search(query_embedding, top_k=5):
    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    
    # Handle the case where no results are found
    if indices.size == 0:
        return []

    results = []
    for i in indices[0]:
        if 0 <= i < len(documents):
            results.append(documents[i])
        else:
            results.append("Invalid index")  # Handle invalid index gracefully
    
    return results
