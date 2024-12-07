import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import os

# Model and tokenizer loading
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, device_map='auto', torch_dtype=torch.float32
)

# File loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    split_texts = [text_splitter.split_text(page.page_content) for page in pages]
    return split_texts, pages

# Summarization for a single page
def summarize_text(text, max_length=500, min_length=50):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=max_length,
        min_length=min_length
    )
    result = pipe_sum(text)
    return result[0]['summary_text']

@st.cache_data
# Function to display the PDF of a given file
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App (Multi-Page Support)")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        # Create 'data/' directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save uploaded file
        filepath = os.path.join("data", uploaded_file.name)
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        
        split_texts, pages = file_preprocessing(filepath)
        
        st.info(f"Uploaded file contains {len(pages)} pages.")
        summary_mode = st.radio("Select Summary Mode", ["Page-wise Summarization", "Entire Document Summarization"])
        
        # Detail level options
        detail_level = st.radio("Select Detail Level", ["Brief", "Detailed"])
        max_length = 300 if detail_level == "Brief" else 500
        min_length = 30 if detail_level == "Brief" else 50

        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("Uploaded File")
                displayPDF(filepath)

            with col2:
                if summary_mode == "Page-wise Summarization":
                    st.info("Page-wise Summaries")
                    for idx, page_texts in enumerate(split_texts):
                        st.subheader(f"Page {idx + 1}")
                        page_summary = summarize_text(" ".join(page_texts), max_length=max_length, min_length=min_length)
                        st.text_area(f"Summary for Page {idx + 1}", value=page_summary, height=150, key=f"page_{idx+1}_summary")
                elif summary_mode == "Entire Document Summarization":
                    st.info("Summarizing Entire Document... Please wait.")
                    combined_text = " ".join([" ".join(page_texts) for page_texts in split_texts])
                    document_summary = summarize_text(combined_text, max_length=max_length, min_length=min_length)
                    st.success("Summarization Complete")
                    st.text_area("Document Summary", value=document_summary, height=300)

if __name__ == "__main__":
    main()
