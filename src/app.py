import streamlit as st
from retriever.retriever import retrieve_docs
from generator.generator import generate_response_with_gemini
from ingestion.text_ingestion import ingest_text

st.title("Vegam Solutions RAG System")

file = st.file_uploader("Upload a file", type=["txt", "png","mp3", "mp4", "csv"])
query = st.text_input("Enter your query here")

if st.button("submit"):
    if file and query:
        st.write("Processing your file...")

        if file.type == "text/plain":  # If the uploaded file is a text file
            # Save the uploaded file to a temporary location
            with open("temp_text_file.txt", "wb") as f:
                f.write(file.getbuffer())
            
            # Call ingest_text to process the text file and add its embeddings to the FAISS index
            documents = ingest_text("temp_text_file.txt")  # Ingest the text file

            # st.write(f"Processed {len(documents)} documents.")
        
        # If the file type is something else (image, audio, etc.), handle accordingly
        elif file.type in ["image/png", "audio/mp3", "video/mp4", "text/csv"]:
            st.error("This file type is not supported for direct ingestion yet.")
            
        
        docs = retrieve_docs(file, query)
        # st.write("Retrieved documents:")
        # st.write(docs)

        response = generate_response_with_gemini(docs, query)
        st.write("Generated response:")
        st.write(response)
    else:
        st.error("Please upload a file and enter a query")