from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from pipeline.state import PipelineState
from utils.pdf import extract_text


class ScoreOutput(BaseModel):
    score: int
    reasoning: str


_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
_score_llm = _llm.with_structured_output(ScoreOutput)


def extract_node(state: PipelineState) -> PipelineState:
    return state


def score_node(state: PipelineState) -> PipelineState:
    prompt = ChatPromptTemplate.from_template(
        "You are an expert recruiter. Given the job description and a candidate's resume, "
        "score how well the candidate matches the role on a scale of 0-100. "
        "Return a score and one concise sentence of reasoning.\n\n"
        "Job Description:\n{jd_text}\n\n"
        "Resume:\n{resume_text}"
    )
    chain = prompt | _score_llm

    updated_resumes = []
    for resume in state["resumes"]:
        result: ScoreOutput = chain.invoke({
            "jd_text": state["jd_text"],
            "resume_text": resume["text"],
        })
        updated_resumes.append({
            **resume,
            "score": result.score,
            "reasoning": result.reasoning,
        })

    return {**state, "resumes": updated_resumes}


def rank_node(state: PipelineState) -> PipelineState:
    ranked = sorted(state["resumes"], key=lambda r: r["score"], reverse=True)
    return {**state, "resumes": ranked}


def question_gen_node(state: PipelineState) -> PipelineState:
    prompt = ChatPromptTemplate.from_template(
        "You are an expert interviewer. Given the job description and this candidate's resume, "
        "generate exactly 5 targeted interview questions specific to this candidate's background "
        "and the role requirements. Return only the 5 questions as a numbered list, one per line.\n\n"
        "Job Description:\n{jd_text}\n\n"
        "Resume:\n{resume_text}"
    )
    chain = prompt | _llm

    updated_resumes = []
    for resume in state["resumes"]:
        result = chain.invoke({
            "jd_text": state["jd_text"],
            "resume_text": resume["text"],
        })
        raw = result.content.strip()
        questions = [
            line.lstrip("0123456789. ").strip()
            for line in raw.splitlines()
            if line.strip()
        ]
        updated_resumes.append({**resume, "questions": questions})

    return {**state, "resumes": updated_resumes}
