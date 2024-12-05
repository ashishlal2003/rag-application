import streamlit as st
import os
from retriever.retriever import retrieve_docs
from generator.generator import generate_response_with_gemini
from ingestion.text_ingestion import ingest_text
from ingestion.pdf_ingestion import ingest_pdf

# Set page configuration
st.set_page_config(
    page_title="Vegam Solutions RAG System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for improved styling
st.markdown("""
    <style>
    .main-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        border: 2px solid #4CAF50;
        border-radius: 8px;
    }
    .response-box {
        background-color: gray;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Main app
def main():
    st.title("🤖 Vegam Solutions RAG System")
    st.markdown("### Retrieve and Generate Insights from Your Documents")

    # Create a container for file upload and query
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            file = st.file_uploader(
                "Upload a file", 
                type=["txt", "pdf"],
                help="Support for txt and pdf files currently available"
            )
        
        with col2:
            st.write("") # Add some spacing
            st.write("") # Add some spacing

        query = st.text_input(
            "Enter your query here", 
            placeholder="Ask a question about your document..."
        )

        # Submit button
        submit_button = st.button("Generate Response", type="primary")

    # Response section
    if submit_button:
        if file and query:
            with st.spinner('Processing your document...'):
                try:
                    # Temporary file handling
                    temp_file_path = os.path.join("temp", file.name)
                    os.makedirs("temp", exist_ok=True)
                    
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Ingest based on file type
                    if file.type == "text/plain":
                        documents = ingest_text(temp_file_path)
                    elif file.type == "application/pdf":
                        documents = ingest_pdf(temp_file_path)
                    
                    # Retrieve and generate response
                    docs = retrieve_docs(file, query)
                    response = generate_response_with_gemini(docs, query)
                    
                    # Display response with custom styling
                    st.markdown("""
                        <div class="response-box">
                            <h3>🤖 Generated Response:</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    st.write(response)
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
        
        else:
            st.warning("Please upload a file and enter a query")

    # Sidebar for additional information
    st.sidebar.title("About")
    st.sidebar.info("""
    ### Vegam Solutions RAG System
    - Upload a text or PDF document
    - Ask a question about the document
    - Get an AI-generated response
    
    **Supported File Types:**
    - Text (.txt)
    - PDF (.pdf)
    """)

if __name__ == "__main__":
    main()