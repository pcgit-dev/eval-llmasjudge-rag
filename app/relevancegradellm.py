#This evaluator checks if the answer addresses the user’s question, without needing a ground 
# truth reference answer.
# we simply look at the inputs and outputs without needing the reference_outputs. 
# Without a reference answer we can't grade accuracy, but can still grade relevance—as in, 
# did the model address the user's question or not.

from langchain_openai import ChatOpenAI
from relevancegrade import RelevanceGrade

class RelevanceGradeEvaluator:
    def __init__(self):
        self.relevance_instructions = """You are a teacher grading a quiz. 

            You will be given a QUESTION and a STUDENT ANSWER. 

            Here is the grade criteria to follow:
            (1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
            (2) Ensure the STUDENT ANSWER helps to answer the QUESTION

            Relevance:
            A relevance value of True means that the student's answer meets all of the criteria.
            A relevance value of False means that the student's answer does not meet all of the criteria.

            Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

            Avoid simply stating the correct answer at the outset."""
            # Grader LLM
        self.relevance_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(RelevanceGrade, method="json_schema", strict=True)

    # Evaluator
    def relevance(self,inputs: dict, outputs: dict) -> bool:
        """A simple evaluator for RAG answer helpfulness."""
        answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = self.relevance_llm.invoke([
            {"role": "system", "content": self.relevance_instructions}, 
            {"role": "user", "content": answer}
        ])
        return grade["relevant"]