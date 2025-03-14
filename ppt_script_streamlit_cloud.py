import streamlit as st
import nltk
import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from nltk.data import find

def load_environment():
    _ = load_dotenv(find_dotenv())
    return os.getenv("GROQ_API_KEY")

def download_nltk_tokenizer():
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def save_uploaded_file(uploaded_file):
    save_path = os.path.join("data", uploaded_file.name)
    os.makedirs("data", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def process_ppt(file_path):
    loader = UnstructuredPowerPointLoader(file_path, mode="elements")
    docs = loader.load()
    ppt_data = {}
    for doc in docs:
        page = doc.metadata["page_number"]
        ppt_data[page] = ppt_data.get(page, "") + "\n\n" + doc.page_content
    context = "".join([f"### Slide {page}:\n\n{content.strip()}\n\n" for page, content in ppt_data.items()])
    return context

def generate_speaker_script(context, api_key):
    question = """
    For each PowerPoint slide provided above, write a 2-minute script that effectively conveys the key points.
    Ensure a smooth flow between slides, maintaining a clear and engaging narrative.
    """
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    system = SystemMessagePromptTemplate.from_template("You are a helpful AI assistant who answers user questions based on the provided context.")
    prompt_template = HumanMessagePromptTemplate.from_template(
        """Answer user question based on the provided context ONLY! If you do not know the answer, just say "I don't know".
        ### Context:
        {context}
        
        ### Question:
        {question}
        
        ### Answer:"""
    )
    messages = [system, prompt_template]
    template = ChatPromptTemplate(messages)
    qna_chain = template | llm | StrOutputParser()
    return qna_chain.invoke({'context': context, 'question': question})

def main():
    st.title("PowerPoint Speaker Script Generator")
    st.write("Upload a PowerPoint file, and the app will generate a speaker script for each slide.")
    
    groq_api_key = load_environment()
    download_nltk_tokenizer()
    uploaded_file = st.file_uploader("Upload a PowerPoint file", type=["pptx"])
    
    if uploaded_file is not None:
        save_path = save_uploaded_file(uploaded_file)
        st.success("File uploaded successfully!")
        
        if st.button("Generate Speaker Script"):
            with st.spinner("Processing... This may take a while."):
                context = process_ppt(save_path)
                response = generate_speaker_script(context, groq_api_key)
                output_path = os.path.join("data", "ppt_script.md")
                with open(output_path, "w") as f:
                    f.write(response)
                
                st.success("Speaker script generated successfully!")
                st.download_button(label="Download Speaker Script", data=response, file_name="ppt_script.md", mime="text/markdown")
                st.markdown("### Generated Script")
                st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>{response}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
