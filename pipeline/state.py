from typing import TypedDict


class ResumeEntry(TypedDict):
    filename: str
    text: str
    score: int
    reasoning: str
    questions: list[str]


class PipelineState(TypedDict):
    jd_text: str
    resumes: list[ResumeEntry]
