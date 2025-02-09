from src.helper import load_pdf_file, text_split, download_embedding
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["PINECONE_API_KEY"] = os.environ.get("PINE_CONE_API_KEY")

extracted_data = load_pdf_file(data="data/")
text_chunks = text_split(extracted_data)
embeddings = download_embedding()

api_key = os.environ.get("PINE_CONE_API_KEY")
pc = Pinecone(api_key= api_key)

index = "testbot"

# pc.create_index(
#     name = index,
#     dimension=384,
#     metric="cosine",
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     )
# )

# docsearch = PineconeVectorStore.from_documents(
#     documents=text_chunks,
#     index_name=index,
#     embedding=embeddings,
# )
