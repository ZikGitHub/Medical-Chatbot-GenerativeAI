from src.helper import load_pdf_file, text_split, download_embedding
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import time
import os

load_dotenv()

os.environ["PINECONE_API_KEY"] = os.environ.get("PINE_CONE_API_KEY")

extracted_data = load_pdf_file(data="data/")
text_chunks = text_split(extracted_data)
embeddings = download_embedding()

api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key= api_key)

index_name = "testbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1"
        ) 
    ) 

# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)