from dotenv import load_dotenv
import os

from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load environment variables
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")


# 1. Hugging Face LLM Setup
HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-7B-Instruct-1M"

client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_TOKEN
)


# 2. Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.

If the answer cannot be found in the provided context, politely say:
"I'm sorry, but I couldn't find information about that in the provided document."

Do not make up an answer or use information outside the provided context.

Context:{context}
Question:{question}

Start the answer directly. No small talk please.
"""


# 3. Load FAISS Database
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


# 4. Ask User for Query
user_query = input("Write Query Here: ")


# 5. Retrieve relevant documents from FAISS
docs = db.similarity_search(
    user_query,
    k=3
)


# 6. Combine retrieved documents into context
context = "\n\n".join(
    doc.page_content
    for doc in docs
)


# 7. Create the final prompt
prompt = CUSTOM_PROMPT_TEMPLATE.format(
    context=context,
    question=user_query
)


# 8. Send context + question to the LLM
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


# 9. Print the answer
answer = response.choices[0].message.content

print("\nRESULT:\n")
print(answer)


# 10. Print the source documents
print("\nSOURCE DOCUMENTS:\n")

for i, doc in enumerate(docs, start=1):
    print(f"--- Source {i} ---")
    print(doc.page_content)
    print()