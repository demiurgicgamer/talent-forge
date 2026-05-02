from typing import TypedDict
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
# State
# -------------------
class ResumeState(TypedDict):
    resume_text: str
    job_description: str
    analysis: str
    fit_score: int
    improved_resume: str
    approved: bool

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
builder.add_node("read_job", read_job)
builder.add_node("analyze_match", analyze_match)
builder.add_node("improve_resume", improve_resume)
builder.add_node("human_approval", human_approval)
builder.add_node("retry_improvement", retry_improvement)

builder.set_entry_point("read_resume")

builder.add_edge("read_resume", "read_job")
builder.add_edge("read_job", "analyze_match")
builder.add_edge("analyze_match", "improve_resume")
builder.add_edge("improve_resume", "human_approval")

builder.add_conditional_edges(
    "human_approval",
    approval_route,
    {
        "end": END,
        "retry": "retry_improvement"
    }
)

builder.add_edge("retry_improvement", "human_approval")

graph = builder.compile()

# -------------------
# Run
# -------------------
result = graph.invoke({})
print(result["analysis"])
print(result["fit_score"])
print(result.get("improved_resume", ""))