import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from augmented import get_augmented_results
from generation import generate_response, save_response

def main():
    # Set up logger
    logger = setup_logger()
    
    question = "ما معنى ODP"
    
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
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()