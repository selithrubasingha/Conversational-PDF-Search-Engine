import os
# loading
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI  # llm
from langchain_openai import OpenAIEmbeddings  # embeddings
from langchain_core.vectorstores import InMemoryVectorStore  # vector_store
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader  # document_loader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # text splitter
from langchain import hub  # hub(like github but just for langchain)
from langchain_core.documents import Document  # tools for making state and nodes
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph  # langgraph

llm = ChatOpenAI(model="gpt-4o-mini")

embeddings = OpenAIEmbeddings()


def document_storage(document):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, "temp")

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    temp_path = os.path.join(temp_dir, document.name)

    try:
        with open(temp_path, "wb") as f:
            f.write(document.getbuffer())
    except Exception as e:
        raise RuntimeError(f"document_storage failed: {e}")

    if not os.path.exists(temp_path):
        return

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        # Your RAG logic would go here. For now, we'll just confirm.

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document

        )

        all_splits = text_splitter.split_documents(docs)
        # vector_store.add_documents(documents = all_splits)

        vector_store = FAISS.from_documents(documents=all_splits, embedding=embeddings)

        return vector_store
    except Exception as e:
        raise RuntimeError(f"document loading failed: {e}")


def document_loader(document, question):
    # Get the absolute path of the directory where the script is running
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, "temp")

    # --- 1. Create the temporary directory if it doesn't exist ---
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # --- 2. Create the full, absolute path for the temporary file ---
    temp_path = os.path.join(temp_dir, document.name)

    # --- 3. Save the uploaded file ---
    try:
        with open(temp_path, "wb") as f:
            f.write(document.getbuffer())
    except Exception as e:
        return  # Stop execution if file saving fails

    # --- 4. Verify the file exists before loading ---
    if not os.path.exists(temp_path):
        return

    # --- 5. Load the document from the verified path ---
    answer = ""
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        # Your RAG logic would go here. For now, we'll just confirm.
        answer = agent_answer(document=docs, question=question)
    except Exception as e:
        # This new line will give you much more detail

        # You might want to stop execution or return an error message here
        return f"Failed to process the PDF. Error: {type(e).__name__}"
    finally:
        # --- 6. Clean up the temporary file ---
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return answer


def agent_answer(vector_store, question):
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # temp_dir = os.path.join(script_dir, "temp")

    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)

    # temp_path = os.path.join(temp_dir, document.name)
    # with open(temp_path, "wb") as f:
    #     f.write(document.getbuffer())

    # file_path = document
    # # # loader = UnstructuredPDFLoader(file_path)
    # loader = UnstructuredPDFLoader(file_path)
    # docs = loader.load()

    # docs = document

    # # assert len(docs) == 1

    # text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=1000,  # chunk size (characters)
    #     chunk_overlap=200,  # chunk overlap (characters)
    #     add_start_index=True,  # track index in original document

    # )

    # all_splits = text_splitter.split_documents(docs)
    # document_ids = vector_store.add_documents(documents = all_splits)

    # vector_store.add_documents(documents = all_splits)

    prompt = hub.pull("rlm/rag-prompt")

    example_messages = prompt.invoke(
        {"context": "(context goes here)", "question": "(question goes here)"}
    ).to_messages()

    assert len(example_messages) == 1

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    result = graph.invoke({"question": question})

    return f"Answer: {result['answer']}"


