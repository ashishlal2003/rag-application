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
    
    # Debugging: Print size and document being added
    print(f"Adding document: {doc} with embedding of shape {embedding.shape}")  # Debugging line
    index.add(embedding)  # Add the embedding to the FAISS index
    documents.append(doc)  # Store the document

def search(query_embedding, top_k=5):
    """
    Perform a similarity search in the FAISS index.
    """
    print(f"Searching for query embedding: {query_embedding[:5]}...")  # Print part of the query embedding for debugging
    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    
    # Debugging: Print search result indices
    print(f"Search result indices: {indices}")
    
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
