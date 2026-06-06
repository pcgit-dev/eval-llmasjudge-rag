from typing_extensions import Annotated,TypedDict

## Relevance Output Schema
class RelevanceGrade(TypedDict):
    # Note that the order in the fields are defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the answer is relevant and helpful to the question, False otherwise."]

