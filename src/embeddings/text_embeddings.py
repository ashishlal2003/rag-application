from sentence_transformers import SentenceTransformer

def text_embeddings(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text)