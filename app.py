from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

# -------------------
# LLM Setup
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
    roles: str

# -------------------
# Node 1 Read Resume
# -------------------
def read_resume(state):
    with open("sample_resume.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("\n Resume Loaded\n")
    return {"resume_text": text}

# -------------------
# Node 2 Analyze Resume
# -------------------
def analyze_resume(state):
    prompt = f"""
    Analyze this resume.
    Extract:
    1. Main skills
    2. Years of experience
    3. Strengths

    Resume:
    {state["resume_text"]}
    """

    result = llm.invoke(prompt)

    print("\n Resume Analyzed\n")
    return {"analysis": result.content}

# -------------------
# Node 3 Suggest Roles
# -------------------
def suggest_roles(state):
    prompt = f"""
    Based on this analysis suggest top 5 job roles.

    Analysis:
    {state["analysis"]}
    """

    result = llm.invoke(prompt)

    print("\n Roles Suggested\n")
    return {"roles": result.content}

# -------------------
# Build Graph
# -------------------
builder = StateGraph(ResumeState)

builder.add_node("read_resume", read_resume)
builder.add_node("analyze_resume", analyze_resume)
builder.add_node("suggest_roles", suggest_roles)

builder.set_entry_point("read_resume")

builder.add_edge("read_resume", "analyze_resume")
builder.add_edge("analyze_resume", "suggest_roles")
builder.add_edge("suggest_roles", END)

graph = builder.compile()

# -------------------
# Run Graph
# -------------------
result = graph.invoke({})

print("\n FINAL OUTPUT")
print("=" * 50)
print(result["roles"])