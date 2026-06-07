# Evaluators or Metrics
# Correctness: Response vs reference answer
# Goal: Measure "how similar/correct is the RAG chain answer, relative to a ground-truth answer"
# Mode: Requires a ground truth (reference) answer supplied through a dataset
# Evaluator: Use LLM-as-judge to assess answer correctness.
# This evaluator checks if the RAG answer is factually accurate compared to the ground truth:
#A key design pattern here is the use of with_structured_output(). 
# By forcing the LLM to produce a typed output (boolean correct field plus an explanation),
#  we get reliable, parseable evaluation results. The explanation field is also valuable — it forces the model
#  to reason through its assessment before providing a score, which improves judgment quality.
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