from typing import TypedDict, List, Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import re

load_dotenv()

# -------------------
# LLM
# -------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# -------------------
# State (FINAL CLEAN)
# -------------------
class ResumeState(TypedDict):
    resume_text: str
    jobs: List[Dict]
    scored_jobs: List[Dict]
    best_job: Dict
    tailored_resume: str
    cover_letter: str
    approved: bool

# -------------------
# Jobs (mock for now)
# -------------------
def load_jobs(state):
    jobs = [
        {"title": "XR Developer", "description": "Unity, AR/VR, C#, AI integration"},
        {"title": "AI Engineer", "description": "LLMs, Python, APIs, LangChain"},
        {"title": "Game Developer", "description": "Unreal, C++, multiplayer systems"},
        {"title": "Simulation Engineer", "description": "3D simulation, physics, XR systems"},
    ]
    return {"jobs": jobs}

# -------------------
# Read Resume (from UI input)
# -------------------
def read_resume(state):
    return {"resume_text": state["resume_text"]}

# -------------------
# Score Jobs
# -------------------
def score_jobs(state):
    results = []

    for job in state["jobs"]:
        prompt = f"""
        Compare resume and job.

        Resume:
        {state["resume_text"]}

        Job:
        {job["title"]} - {job["description"]}

        Return:
        Score (0-100)
        Reason
        """

        response = llm.invoke(prompt).content

        match = re.search(r"(\d{1,3})", response)
        score = int(match.group(1)) if match else 50

        results.append({
            "title": job["title"],
            "description": job["description"],
            "score": score,
            "reason": response
        })

    return {"scored_jobs": results}

# -------------------
# Select Best Job
# -------------------
def select_best_job(state):
    sorted_jobs = sorted(
        state["scored_jobs"],
        key=lambda x: x["score"],
        reverse=True
    )

    return {"best_job": sorted_jobs[0]}

# -------------------
# Generate Resume
# -------------------
def generate_resume(state):
    job = state["best_job"]

    prompt = f"""
    Rewrite resume for this job:

    Job: {job["title"]}
    Description: {job["description"]}

    Resume:
    {state["resume_text"]}

    Make ATS optimized and keyword rich.
    """

    result = llm.invoke(prompt)

    return {"tailored_resume": result.content}

# -------------------
# Cover Letter
# -------------------
def generate_cover_letter(state):
    job = state["best_job"]

    prompt = f"""
    Write cover letter for:

    Job: {job["title"]}

    Based on resume:
    {state["tailored_resume"]}
    """

    result = llm.invoke(prompt)

    return {"cover_letter": result.content}

# -------------------
# Human Approval (UI handles this later)
# -------------------
def auto_approve(state):
    return {"approved": True}

# -------------------
# Router
# -------------------
def approval_route(state):
    if state["approved"]:
        return "end"
    return "retry"

# -------------------
# Retry
# -------------------
def retry_generation(state):
    prompt = f"""
    Improve resume and cover letter.
    Make it stronger and more professional.
    """

    result = llm.invoke(prompt)

    return {
        "tailored_resume": result.content,
        "cover_letter": result.content
    }

# -------------------
# Graph
# -------------------
builder = StateGraph(ResumeState)

builder.add_node("read_resume", read_resume)
builder.add_node("load_jobs", load_jobs)
builder.add_node("score_jobs", score_jobs)
builder.add_node("select_best_job", select_best_job)
builder.add_node("generate_resume", generate_resume)
builder.add_node("generate_cover_letter", generate_cover_letter)
builder.add_node("auto_approve", auto_approve)
builder.add_node("retry_generation", retry_generation)

builder.set_entry_point("read_resume")

builder.add_edge("read_resume", "load_jobs")
builder.add_edge("load_jobs", "score_jobs")
builder.add_edge("score_jobs", "select_best_job")
builder.add_edge("select_best_job", "generate_resume")
builder.add_edge("generate_resume", "generate_cover_letter")
builder.add_edge("generate_cover_letter", "auto_approve")

builder.add_conditional_edges(
    "auto_approve",
    approval_route,
    {
        "end": END,
        "retry": "retry_generation"
    }
)

builder.add_edge("retry_generation", "auto_approve")

graph = builder.compile()

# -------------------
# EXPOSE FUNCTION FOR STREAMLIT
# -------------------
def run_graph(resume_text: str):
    return graph.invoke({
        "resume_text": resume_text
    })