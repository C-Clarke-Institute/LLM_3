import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc.create_index(
        name="hrdocumentsv3",
        vector_type="dense",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled",
        tags={
            "environment": "development"
        }
    )
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
loader = PyPDFLoader("hrdocument.pdf", mode="page")
all_splits = text_splitter.split_documents(loader.load())
vector_store = PineconeVectorStore(
    index_name="hrdocumentsv3",
    embedding=embedding_model,
    namespace="hr-document"
)
vector_store.add_documents(documents=all_splits)