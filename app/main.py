# Example usage
import os

from raggeneration import raggeneration
from dotenv import load_dotenv

if __name__ == "__main__":
   
    load_dotenv()

    os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
    os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
    os.environ["LANGSMITH_TRACING"]="true"
    rag = raggeneration()
    rag.rag_uploader()
    response = rag.rag_bot("What is Agents?")
    print(response)
