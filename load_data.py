import os 
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import SelfHostedEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv 
from embeddings import StellaEmbeddings

load_dotenv()
# -----------------------------------------------------------------------------
# Configuration: parameterize model and data file paths via environment variables
#   STELLA_MODEL_PATH: path or identifier for the SentenceTransformer model
#   MOVIES_JSON_PATH: path to the JSON file containing movie content and metadata
# Defaults: model "dunzhang/stella_en_400M_v5" and "movies_content_metadata.json" in CWD
# -----------------------------------------------------------------------------
MODEL_PATH = os.getenv("STELLA_MODEL_PATH", "dunzhang/stella_en_400M_v5")
DATA_JSON_PATH = os.getenv("MOVIES_JSON_PATH", "movies_content_metadata.json")

ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
embeddings = StellaEmbeddings(model_path=MODEL_PATH)
vectorstore = AstraDBVectorStore(
            embedding=embeddings,
            collection_name="movies_stella_ft",
            token=ASTRA_DB_APPLICATION_TOKEN,
            api_endpoint=ASTRA_DB_ENDPOINT,
        )

import json
# Load movie data from JSON file (path configurable via MOVIES_JSON_PATH)
with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
    movies_data = json.load(f)

# Create LangChain Document objects
movies = [
    Document(page_content=entry.get("content", ""), metadata=entry.get("metadata", {}))
    for entry in movies_data
]
vectorstore.add_documents(movies)


