from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.components.llm import get_llm
from app.components.vector_store import load_vector_store
from app.config.config import HUGGINGFACE_REPO_ID, HUGGINGFACEHUB_API_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """You are a helpful and friendly medical assistant. 
- If the user question is **medical-related**, answer in 2–3 sentences using only the information provided in the context. 
- If the answer is not in the context, say: "I don’t know based on the provided context."
- If the user question is **general or conversational (like greetings, small talk, how are you, etc.)**, respond politely and naturally without needing the context.

Context:
{context}

Question:
{question}

Answer:
"""



def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE, 
        input_variables = ['context', 'question']
        )

def create_qa_chain():
    try:
        logger.info("loading vector store for context")
        
        db = load_vector_store()
        if db is None:
            raise CustomException("Vector store not present")
        
        llm = get_llm()
        
        if llm is None:
            raise CustomException("LLM not present")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm = llm, 
            chain_type = "stuff", 
            retriever = db.as_retriever(search_kwargs = {"k": 1}), 
            return_source_documents = True, 
            chain_type_kwargs = {"prompt": set_custom_prompt()})
        
        logger.info("QA chain created successfully")
        return qa_chain
    except Exception as e:
        error_message = CustomException("Failed to create QA chain", e)
        logger.error(str(error_message))
        raise error_message
