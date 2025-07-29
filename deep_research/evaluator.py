from agents import Agent
from pydantic import BaseModel, Field

INSTRUCTIONS = (
    "You are an evaluation agent that assesses the quality of research results. "
    "You should evaluate whether the search results are comprehensive, relevant, and sufficient "
    "to answer the research query. Consider factors like: "
    "- Relevance to the original query "
    "- Breadth and depth of information "
    "- Quality and reliability of sources "
    "- Completeness of coverage "
    "Provide a clear assessment with reasoning."
)

class Evaluator(BaseModel):
    is_good: bool = Field(description="Whether the search results are good enough to proceed with report writing")
    reason: str = Field(description="Detailed reasoning for the evaluation, including strengths and weaknesses")

evaluator = Agent(
    name="Evaluator",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=Evaluator
)