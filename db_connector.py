import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import load_config
from dotenv import load_dotenv
from pymongo import MongoClient
from urllib.parse import quote_plus
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()
def vector_search():
    config = load_config()

    # Initialize embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=config["model"]["embedding_model_name"],
        task_type=config["model"]["embedding_task_type"]
    )


    # MongoDB connection setup
    username = os.getenv("MONGODB_USERNAME")
    password = os.getenv("MONGODB_PASSWORD")
    escaped_username = quote_plus(username)
    escaped_password = quote_plus(password)
    MONGODB_ATLAS_CLUSTER_URI = f"mongodb+srv://{escaped_username}:{escaped_password}@{config['mongodb']['cluster_uri']}/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI, tlsAllowInvalidCertificates=True)
    DB_NAME = config["mongodb"]["db_name"]
    COLLECTION_NAME = config["mongodb"]["collection_name"]
    ATLAS_VECTOR_SEARCH_INDEX_NAME = config["mongodb"]["index_name"]

    # Access the MongoDB collection
    MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

    # Initialize vector store
    vector_store = MongoDBAtlasVectorSearch(
        embedding=embedding_model,
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        relevance_score_fn="cosine",
    )
    return vector_store