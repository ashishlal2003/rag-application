from PyPDF2 import PdfReader
from embeddings.pdf_embeddings import pdf_embeddings
from vector.vector_store import add_to_index

def ingest_pdf(file_path):
    extracted_text = []

    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text.append(page_text)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []
    
    for chunk in extracted_text:
        embedding = pdf_embeddings(chunk)
        add_to_index(embedding, chunk)

    return extracted_text
