from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from get_unique_doc import get_pairs
import logging
from augmented import get_augmented_results
from operator import itemgetter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Define prompt template for generating the final response
answer_template = """أجب على السؤال التالي بناءً على هذا السياق:

{context}

السؤال: {question}
"""

prompt_generation = ChatPromptTemplate.from_template(answer_template)

def generate_response(question, documents):
    """Generate a response based on the retrieved documents."""
    try:
        reRanked_docs = get_pairs(question, documents)
        final_rag_chain = (
            {"context": itemgetter("context"), 
            "question": itemgetter("question")} 
            | prompt_generation
            | llm
            | StrOutputParser()
        )
        return final_rag_chain.invoke({
            "context": reRanked_docs, 
            "question": question
        })
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise

def save_response(response, filename='response.txt'):
    """Save the generated response to a file."""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(response)
        logger.info(f"Response saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving response: {e}")
        raise

# def main():
#     question = "ما معنى ODP"
    
#     try:
#         # Step 1: Get augmented results (query perspectives and documents)
#         documents = get_augmented_results(question)
        
#         # Step 2: Generate a response based on the retrieved documents
#         response = generate_response(question, documents)
#         logger.info("Response generated successfully.")
        
#         # Step 3: Save the response to a file
#         save_response(response)
        
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()