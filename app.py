import os
import streamlit as st

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load environment variables
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")

HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-7B-Instruct-1M"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load FAISS vector database
@st.cache_resource
def get_vectorstore():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db

# Create Hugging Face client
@st.cache_resource
def get_llm_client():

    client = InferenceClient(
        provider="featherless-ai",
        api_key=HF_TOKEN
    )

    return client


# Custom prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.

If the answer cannot be found in the provided context, politely say:
"I'm sorry, but I couldn't find information about that in the provided document."

Do not make up an answer or use information outside the provided context.

Context:
{context}

Question:
{question}

Start the answer directly. No small talk please.
"""

# Generate answer
def get_answer(user_query):

    # Load vector database
    db = get_vectorstore()

    # Retrieve relevant documents
    docs = db.similarity_search(
        user_query,
        k=3
    )

    # Combine retrieved text
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # Create prompt
    prompt = CUSTOM_PROMPT_TEMPLATE.format(
        context=context,
        question=user_query
    )

    # Get LLM client
    client = get_llm_client()

    # Ask Qwen
    response = client.chat_completion(
        model=HUGGINGFACE_REPO_ID,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=512,
        temperature=0.5
    )

    answer = response.choices[0].message.content

    return answer, docs


# Streamlit UI
def main():

    st.title("Medical RAG Chatbot")

    st.write(
        "Ask questions based on the information available "
        "in the medical document."
    )

    # Store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_query = st.chat_input(
        "Ask a medical question..."
    )

    if user_query:

        # Display user question
        with st.chat_message("user"):
            st.markdown(user_query)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query
            }
        )

        # Generate answer
        try:

            with st.spinner("Searching the document..."):

                answer, source_documents = get_answer(
                    user_query
                )

            # Display answer
            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            # Display sources
            with st.expander("View source documents"):

                for i, doc in enumerate(
                    source_documents,
                    start=1
                ):

                    st.markdown(
                        f"**Source {i}**"
                    )

                    st.write(
                        doc.page_content
                    )

        except Exception as e:

            st.error(
                f"Something went wrong: {str(e)}"
            )


if __name__ == "__main__":
    main()