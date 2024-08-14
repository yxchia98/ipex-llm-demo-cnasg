# import
from llama_index.core import StorageContext, Settings, PromptTemplate, VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.ipex_llm import IpexLLM
from llama_index.embeddings.ipex_llm import IpexLLMEmbedding
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url
import psycopg2
import time
import gradio as gr
from tqdm import tqdm
import shutil
import os
from pathlib import Path
import glob


connection_string = f"postgresql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@{os.environ['DB_URL']}:{os.environ['DB_PORT']}?sslmode={os.environ['SSL_MODE']}"
db_name = os.environ['DB_NAME']
conn = psycopg2.connect(connection_string)
# conn = psycopg2.connect(user=os.environ['DB_USER'], password=os.environ['DB_PASSWORD'], host=os.environ['DB_URL'], port=os.environ['DB_PORT'], sslmode=os.environ['SSL_MODE'], connect_timeout=10)
conn.autocommit = True
with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")

url = make_url(connection_string)
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="rag_data",
    embed_dim=384,  # bge-small-en-v1.5 embedding dimension
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)


documents = SimpleDirectoryReader("./data/").load_data()
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
query_engine = index.as_query_engine(streaming=True, similarity_top_k=2)
use_rag = False