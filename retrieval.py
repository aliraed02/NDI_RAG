import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import load_config
from db_connector import vector_search


config = load_config()
vector_store = vector_search()

# Initialize retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": config["retriever"]["k"]}
)


