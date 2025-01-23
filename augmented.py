from operator import itemgetter
from retrieval import retriever
from get_unique_doc import get_unique_union, get_unique_contents
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Define prompt template for generating query perspectives
translate_template = """
أنت مساعد نموذج لغة ذكاء اصطناعي. مهمتك هي إنشاء خمس صيغ مختلفة لسؤال المستخدم لاسترداد المستندات ذات الصلة من قاعدة بيانات ناقلات. من خلال إنشاء وجهات نظر متعددة للسؤال، هدفك هو مساعدة المستخدم على تحسين نتائج البحث وتجاوز قيود البحث القائم على التشابه التقليدي.  
قدم هذه الأسئلة البديلة بشكل واضح ومفصول بأسطر جديدة.  
السؤال الأصلي: {question}
"""

prompt_perspectives = ChatPromptTemplate.from_template(translate_template)

def generate_query_perspectives(question):
    """Generate multiple query perspectives for the given question."""
    try:
        perspectives_chain = (
            {"question": itemgetter("question")}
            | prompt_perspectives 
            | llm
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )
        return perspectives_chain.invoke({"question": question})
    except Exception as e:
        logger.error(f"Error generating query perspectives: {e}")
        raise

def retrieve_documents(perspectives):
    """Retrieve documents based on the generated perspectives."""
    try:
        retrieval_chain = (
            retriever.map()
            | get_unique_union
            | get_unique_contents
        )
        return retrieval_chain.invoke(perspectives)
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise

def get_augmented_results(question):
    """Generate query perspectives and retrieve documents."""
    try:
        # Step 1: Generate multiple query perspectives
        perspectives = generate_query_perspectives(question)
        logger.info("Query perspectives generated successfully.")
        
        # Step 2: Retrieve documents based on the perspectives
        documents = retrieve_documents(perspectives)
        logger.info("Documents retrieved successfully.")
        
        return documents
    except Exception as e:
        logger.error(f"An error occurred in augmented processing: {e}")
        raise