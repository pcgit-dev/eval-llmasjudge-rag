from __future__ import annotations
import logging
from langsmith import Client
from config import get_settings

class dataingetion:
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.log_config()
        self.client = Client(api_key=self.settings.langsmith_api_key)

    def _mask(self, secret: str | None) -> str:
        """Return a safe, partially-masked view of a secret for logging."""
        if not secret:
            return "<not set>"
        return f"{secret[:8]}...{secret[-4:]}" if len(secret) > 12 else "********"

    def log_config(self) -> None:
        """Log the loaded configuration (secrets masked) for sanity checking."""
        s = self.settings
        self.logger.info("OPENAI_API_KEY    : %s", self._mask(s.openai_api_key))
        self.logger.info("GROQ_API_KEY      : %s", self._mask(s.groq_api_key))
        self.logger.info("LANGSMITH_API_KEY : %s", self._mask(s.langsmith_api_key))
        self.logger.info("LANGSMITH_PROJECT : %s", s.langsmith_project)

    def prepare_rag_evaldata(self):
      # Define the examples for the dataset
        examples = [
            {
                "inputs": {"question": "How does the ReAct agent use self-reflection? "},
                "outputs": {"answer": "ReAct integrates reasoning and acting, performing actions - such tools like Wikipedia search API - and then observing / reasoning about the tool outputs."},
            },
            {
                "inputs": {"question": "What are the types of biases that can arise with few-shot prompting?"},
                "outputs": {"answer": "The biases that can arise with few-shot prompting include (1) Majority label bias, (2) Recency bias, and (3) Common token bias."},
            },
            {
                "inputs": {"question": "What are five types of adversarial attacks?"},
                "outputs": {"answer": "Five types of adversarial attacks are (1) Token manipulation, (2) Gradient based attack, (3) Jailbreak prompting, (4) Human red-teaming, (5) Model red-teaming."},
            }
        ]
            ### create the daatset and example in LAngsmith
        dataset_name="RAG Test Evaluation"
        dataset = self.client.create_dataset(dataset_name=dataset_name)
        self.client.create_examples(
                dataset_id=dataset.id,
                examples=examples
       )
