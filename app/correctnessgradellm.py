from langchain_openai import ChatOpenAI
from correctnessgrade import CorrectnessGrade

class CorrectnessGradeEvaluator:
    def __init__(self):
        self.correctness_instructions = """You are a teacher grading a quiz. 

            You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

            Here is the grade criteria to follow:
            (1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
            (2) Ensure that the student answer does not contain any conflicting statements.
            (3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

            Correctness:
            A correctness value of True means that the student's answer meets all of the criteria.
            A correctness value of False means that the student's answer does not meet all of the criteria.

            Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

            Avoid simply stating the correct answer at the outset."""
            # Grader LLM
        self.correctness_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(CorrectnessGrade, method="json_schema", strict=True)

    ## evaluator
    def correctness(self, inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
        """An evaluator for RAG answer accuracy"""
        answers = f"""\
    QUESTION: {inputs['question']}
    GROUND TRUTH ANSWER: {reference_outputs['answer']}
    STUDENT ANSWER: {outputs['answer']}"""

        # Run evaluator
        grade = self.correctness_llm.invoke([
            {"role": "system", "content": self.correctness_instructions}, 
            {"role": "user", "content": answers}
        ])
        return grade["correct"]