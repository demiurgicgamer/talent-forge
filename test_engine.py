from engine import run_graph

resume = """
XR Developer with 8 years experience in Unity, Unreal Engine,
C#, AR/VR systems, AI integration, multiplayer systems.
"""

result = run_graph(resume)

print("\n===== BEST JOB =====\n")
print(result["best_job"])

print("\n===== RESUME =====\n")
print(result["tailored_resume"])

print("\n===== COVER LETTER =====\n")
print(result["cover_letter"])