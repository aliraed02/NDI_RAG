import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from augmented import get_augmented_results
from generation import generate_response, save_response

# Set up logger
logger = setup_logger()

# Define the FastAPI app
app = FastAPI()

# Define a Pydantic model for the request body
class QuestionRequest(BaseModel):
    question: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG API!"}

# Generate response endpoint
@app.post("/generate-response/")
async def generate_response_api(request: QuestionRequest):
    question = request.question
    
    try:
        # Step 1: Get augmented results (query perspectives and documents)
        logger.info("Retrieving augmented results...")
        documents = get_augmented_results(question)
        
        # Step 2: Generate a response based on the retrieved documents
        logger.info("Generating response...")
        response = generate_response(question, documents)
        logger.info("Response generated successfully.")
        
        # Step 3: Save the response to a file
        logger.info("Saving response...")
        save_response(response)
        logger.info("Response saved successfully.")
        
        return {"response": response}
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# To run the API, use the following command:
# uvicorn api:app --reload