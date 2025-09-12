import os
from dotenv import load_dotenv
from langchain_together import Together
from langchain_huggingface import HuggingFaceEndpoint
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import HUGGINGFACE_REPO_ID

logger = get_logger(__name__)
load_dotenv()

def get_llm():
    try:
        if "mistralai" in HUGGINGFACE_REPO_ID.lower():
            if not os.getenv("TOGETHER_API_KEY"):
                raise CustomException("TOGETHER_API_KEY is missing in .env")

            logger.info(f"Loading LLM from Together: {HUGGINGFACE_REPO_ID}")
            return Together(
                model=HUGGINGFACE_REPO_ID,
                temperature=0.5,
                max_tokens=256
            )

        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise CustomException("HUGGINGFACEHUB_API_TOKEN is missing in .env")

        logger.info(f"Loading Hugging Face model: {HUGGINGFACE_REPO_ID}")
        return HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            task="text-generation",
            temperature=0.5,
            max_new_tokens=256,
            return_full_text=False,
            huggingfacehub_api_token=hf_token
        )

    except Exception as e:
        logger.error(f"Error loading LLM: {e}")
        raise CustomException(f"Error loading LLM: {e}")
