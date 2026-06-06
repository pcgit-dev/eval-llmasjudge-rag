from http import client

import openai
from langsmith import Client, wrappers
from correctnessgradellm import CorrectnessGradeEvaluator
from groundedgradellm import GroundedGradeEvaluator
from relevancegradellm import RelevanceGradeEvaluator
from retrievalrelevancegradellm import RetrievalRelevanceGradeEvaluator
from raggeneration import raggeneration
from langsmith import Client
from config import get_settings
class bindeval:
    def __init__(self):
        self.name = "RAG Evaluation"
        self.raggeneration = raggeneration()
        self.raggeneration.rag_uploader()  # build the vector store / retriever before evaluating
        self.correctness_evaluator = CorrectnessGradeEvaluator()
        self.groundedness_evaluator = GroundedGradeEvaluator()
        self.relevance_evaluator = RelevanceGradeEvaluator()
        self.retrieval_relevance_evaluator = RetrievalRelevanceGradeEvaluator()
        self.instructions="Respond to the users question in a short, concise manner (one short sentence)."
        self.model = "gpt-4-turbo" 
        self.settings = get_settings()
        self.client = Client(api_key=self.settings.langsmith_api_key)
    
    ### Call my_app for every datapoints
    def target(self,inputs: dict) -> dict:
        return self.raggeneration.rag_bot(inputs["question"])
    
    def evaluaterag(self) :
        ## Run our evaluation
       self.client.evaluate(
        self.target,
        data="RAG Test Evaluation",
        evaluators=[self.correctness_evaluator.correctness, self.groundedness_evaluator.groundedness, self.relevance_evaluator.relevance, self.retrieval_relevance_evaluator.retrieval_relevance],
        experiment_prefix="rag-doc-relevance",
        metadata={"version": "LCEL context, gpt-4-0125-preview"},
        )
       
