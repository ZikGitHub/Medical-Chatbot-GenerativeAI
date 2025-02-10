from flask import Flask, render_template, request, jsonify
from src.helper import download_embedding
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddings = download_embedding()

index_name = "testbot"  
docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = ChatOllama(model="llama2", temperature=0.4, max_tokens=600)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("response: ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)