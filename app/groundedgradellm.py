from langchain_openai import ChatOpenAI
from groundedgrade import GroundedGrade

class GroundedGradeEvaluator:
    def __init__(self):
        self.grounded_instructions = """You are a teacher grading a quiz. 

            You will be given FACTS and a STUDENT ANSWER. 

            Here is the grade criteria to follow:
            (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
            (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

            Grounded:
            A grounded value of True means that the student's answer meets all of the criteria.
            A grounded value of False means that the student's answer does not meet all of the criteria.

            Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

            Avoid simply stating the correct answer at the outset."""
            # Grader LLM
        self.grounded_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(GroundedGrade, method="json_schema", strict=True)

    # Evaluator
    def groundedness(self, inputs: dict, outputs: dict) -> bool:
        """A simple evaluator for RAG answer groundedness."""
        doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
        answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = self.grounded_llm.invoke([{"role": "system", "content": self.grounded_instructions}, {"role": "user", "content": answer}])
        return grade["grounded"]