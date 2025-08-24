import streamlit as st
from RAG_main import agent_answer, document_loader, document_storage

st.set_page_config(page_title="PDF Chat", page_icon="💬")
st.title("💬 Chat with Your PDF")


@st.cache_resource
def vector_storage(document):
    return document_storage(document)


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Insert Document")
    uploaded_file = st.file_uploader("Upload a PDF and start asking questions", type="pdf")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uploaded_file is not None:
    embedded_pdf = vector_storage(uploaded_file)

    prompt = st.chat_input("Ask something about your document...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Call your RAG function
                response = agent_answer(vector_store=embedded_pdf, question=prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please upload a document to begin the chat.")