
-----

# 📄 Conversational PDF Search Engine

This project is a powerful Q\&A chatbot that allows you to have a conversation with your PDF documents. Instead of manually searching through lengthy files, you can simply upload a document and ask questions in natural language. The application leverages a sophisticated Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers.

*(**Note**: You can replace this with a screenshot of your own app)*

-----

## 🚀 Features

  * **Interactive Chat Interface**: A user-friendly GUI built with Streamlit for a seamless conversational experience.
  * **Dynamic File Upload**: Upload any PDF document on the fly.
  * **Semantic Search**: Goes beyond simple keyword matching to understand the *meaning* behind your questions.
  * **Context-Aware Answers**: The LLM generates responses based *only* on the information present in the uploaded document.
  * **Stateful Agent Logic**: Built with LangGraph to create a robust and reliable RAG chain.

-----

## 🔧 Tech Stack

This project is built with a modern stack for AI application development:

  * **Application Framework**: **Streamlit** - For creating and serving the interactive web GUI.
  * **Orchestration Framework**: **LangChain** - The core framework for building applications with LLMs.
  * **Agent Framework**: **LangGraph** - For creating robust, stateful, and cyclical agent architectures.
  * **Vector Store**: **FAISS** - For efficient storage and similarity search of vector embeddings.
  * **LLM & Embeddings**: **OpenAI** - Used for generating answers and creating text embeddings.

-----

## 🧠 Core Concepts Implemented

This project was a practical application of several key concepts in modern AI:

  * **Retrieval-Augmented Generation (RAG)**: The fundamental architecture of this application. Instead of relying solely on an LLM's internal knowledge, the RAG pipeline first **retrieves** relevant context from the user-provided PDF and then passes that context to the LLM to **generate** an informed answer. This prevents hallucinations and ensures the answers are grounded in the source document.

  * **Vector Embeddings**: Text from the PDF is converted into numerical representations called embeddings. These vectors capture the semantic meaning of the text, allowing the system to find chunks of text that are conceptually similar to the user's question, not just those that share the same keywords.

  * **Vector Stores**: The generated embeddings are stored and indexed in a high-performance vector database (FAISS). This allows for incredibly fast and efficient similarity searches, making the retrieval step of the RAG pipeline possible in real-time.

-----

## 🛠️ Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

  * Python 3.8+
  * An OpenAI API Key

### Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    (First, ensure you have generated a `requirements.txt` file by running `pip freeze > requirements.txt` in your activated environment)

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a file named `.env` in the root of the project directory and add your OpenAI API key:

    ```
    OPENAI_API_KEY="sk-..."
    ```

5.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    The application should now be running in your web browser\!