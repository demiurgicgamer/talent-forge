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
    analysis: str
    score: int
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
def analyze_resume(state):
    prompt = f"""
    Analyze this resume and give:
    Strengths
    Weaknesses
    Score out of 100

    Resume:
    {state["resume_text"]}

    Format:
    Score: number
    Strengths: ...
    Weaknesses: ...
    """

    result = llm.invoke(prompt)
    text = result.content

    match = re.search(r"Score:\s*(\d+)", text)
    score = int(match.group(1)) if match else 50

    return {
        "analysis": text,
        "score": score
    }

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

# -------------------
# Weak Resume Path
# -------------------
def improve_resume(state):
    prompt = f"""
    This resume scored low.

    Suggest how to improve it for better job opportunities.

    Resume:
    {state["resume_text"]}
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
builder.add_node("analyze_resume", analyze_resume)
builder.add_node("suggest_roles", suggest_roles)
builder.add_node("improve_resume", improve_resume)

builder.set_entry_point("read_resume")

builder.add_edge("read_resume", "analyze_resume")

builder.add_conditional_edges(
    "analyze_resume",
    route_resume,
    {
        "strong": "suggest_roles",
        "weak": "improve_resume"
    }
)

builder.add_edge("suggest_roles", END)
builder.add_edge("improve_resume", END)

graph = builder.compile()

# -------------------
# Run
# -------------------
result = graph.invoke({})

print("\nRESUME SCORE:", result["score"])
print("\nANALYSIS:\n")
print(result["analysis"])

print("\nFINAL RESULT:\n")
print(result["result"])