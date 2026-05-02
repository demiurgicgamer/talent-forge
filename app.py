from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import re

load_dotenv()

# -------------------
# Sample Jobs for Testing
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
# LLM
# -------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# -------------------
# State
# -------------------
class ResumeState(TypedDict):
    resume_text: str
    jobs: list
    scored_jobs: list
    ranked_jobs: list
    result: str

# -------------------
# Node 1
# -------------------
def read_resume(state):
    with open("sample_resume.txt", "r", encoding="utf-8") as f:
        text = f.read()

    return {"resume_text": text}

# -------------------
# Node 2
# -------------------
def analyze_match(state):
    prompt = f"""
    Compare Resume and Job Description.

    Output:
    1. Match strengths
    2. Missing skills
    3. Fit score (0-100)

    Resume:
    {state["resume_text"]}

    Job:
    {state["job_description"]}
    """

    result = llm.invoke(prompt)
    text = result.content

    # extract score
    import re
    match = re.search(r"(\d{1,3})", text)
    score = int(match.group(1)) if match else 50

    return {
        "analysis": text,
        "fit_score": score
    }

# -------------------
# Node 3 Job Reader
# -------------------
def read_job(state):
    with open("job.txt", "r", encoding="utf-8") as f:
        job = f.read()

    return {"job_description": job}
# -------------------
# Strong Resume Path
# -------------------
def suggest_roles(state):
    prompt = f"""
    Based on this resume suggest top 5 high-paying matching job roles.

    {state["resume_text"]}
    """

    result = llm.invoke(prompt)

    return {"result": result.content}

#--------------------
#Resume Improvement
#--------------------
def improve_resume(state):
    prompt = f"""
    Improve this resume for the job.

    Resume:
    {state["resume_text"]}

    Job:
    {state["job_description"]}

    Focus:
    - ATS optimization
    - keyword matching
    - clarity
    """

    result = llm.invoke(prompt)

    return {"improved_resume": result.content}
#-------------------
#Score Each Job
#-------------------
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

        import re
        match = re.search(r"(\d{1,3})", response)
        score = int(match.group(1)) if match else 50

        results.append({
            "title": job["title"],
            "score": score,
            "reason": response
        })

    return {"scored_jobs": results}
#---------------------
#Rank Jobs
#---------------------
def rank_jobs(state):
    sorted_jobs = sorted(
        state["scored_jobs"],
        key=lambda x: x["score"],
        reverse=True
    )

    top_jobs = sorted_jobs[:3]

    result_text = "\n".join([
        f"{j['title']} - {j['score']}"
        for j in top_jobs
    ])

    return {
        "ranked_jobs": top_jobs,
        "result": result_text
    }
#---------------------
#Human Approval
#---------------------
def human_approval(state):
    print("\n--- IMPROVED RESUME ---\n")
    print(state["improved_resume"])

    choice = input("\nApprove this resume? (yes/no): ")

    return {
        "approved": choice.lower() == "yes"
    }
#-------------------
#Router Logic
# -------------------
def approval_route(state):
    if state["approved"]:
        return "end"
    return "retry"
#-------------------
# Retry Improvement
#-------------------
def retry_improvement(state):
    prompt = f"""
    Improve again. User rejected previous version.

    Make it more:
    - clear
    - stronger impact
    - better ATS keywords

    Resume:
    {state["resume_text"]}

    Job:
    {state["job_description"]}
    """

    result = llm.invoke(prompt)

    return {"improved_resume": result.content}
# -------------------
# Weak Resume Path
# -------------------
def improve_weak_resume(state):
    prompt = f"""
    This resume scored low.

    Suggest how to improve it for better job opportunities.

    Resume:
    {state["resume_text"]}
    """

    result = llm.invoke(prompt)

    return {"result": result.content}

# -------------------
# Decision Logic
# -------------------
def route(state):
    if state["fit_score"] >= 70:
        return "good_fit"
    return "bad_fit" 
# -------------------
# Good Fit Path
# -------------------  
def improve_for_job(state):
    prompt = f"""
    Improve this resume for the job.

    Resume:
    {state["resume_text"]}

    Job:
    {state["job_description"]}
    """

    result = llm.invoke(prompt)

    return {"result": result.content}

# -------------------
# Router
# -------------------
def route_resume(state):
    if state["score"] >= 75:
        return "strong"
    return "weak"

# -------------------
# Graph
# -------------------
builder = StateGraph(ResumeState)

builder.add_node("read_resume", read_resume)
builder.add_node("load_jobs", load_jobs)
builder.add_node("score_jobs", score_jobs)
builder.add_node("rank_jobs", rank_jobs)

builder.set_entry_point("read_resume")

builder.add_edge("read_resume", "load_jobs")
builder.add_edge("load_jobs", "score_jobs")
builder.add_edge("score_jobs", "rank_jobs")
builder.add_edge("rank_jobs", END)

graph = builder.compile()

# -------------------
# Run
# -------------------
result = graph.invoke({})

print("\nTOP MATCHING JOBS:\n")
print(result["result"])