import streamlit as st
from engine import run_graph, extract_text_from_pdf
import pandas as pd

st.set_page_config(page_title="Career Copilot AI", layout="wide")

st.title("🧠 Career Copilot AI")
st.write("Upload your resume (PDF) or paste text to get AI job matching, resume tuning, and cover letter generation.")

# -------------------
# INPUT SECTION
# -------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📄 Upload Resume PDF", type=["pdf"])

with col2:
    resume_text = st.text_area("✍️ Or Paste Resume Text", height=250)

final_resume = None

if uploaded_file is not None:
    final_resume = extract_text_from_pdf(uploaded_file)
    st.success("PDF loaded successfully")

elif resume_text:
    final_resume = resume_text

# -------------------
# RUN BUTTON
# -------------------
if st.button("🚀 Run AI Career Agent"):

    if not final_resume:
        st.error("Please upload a PDF or paste resume text")

    else:
        with st.spinner("Analyzing resume with AI agent..."):

            result = run_graph(final_resume)

        st.success("Analysis complete!")

        # -------------------
        # BEST JOB (CLEAN UI)
        # -------------------
        job = result.get("best_job", {})

        st.subheader("🏆 Best Job Match")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"### {job.get('title', 'N/A')}")
            st.write(job.get("description", "No description available"))

        with col2:
            st.metric("Overall Score", job.get("score", 0))

        details = job.get("details", {})

        st.write("### 📊 Score Breakdown")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Skills", details.get("skill_match", 0))
        c2.metric("Domain", details.get("domain_match", 0))
        c3.metric("Tools", details.get("tools_match", 0))
        c4.metric("Experience", details.get("experience_match", 0))

        if details.get("summary"):
            st.info(details["summary"])

        # -------------------
        # RANKED JOBS TABLE
        # -------------------
        st.subheader("📊 Job Rankings")

        ranked = result.get("ranked_jobs", [])

        if ranked:
            df = pd.DataFrame([
                {
                    "Job Title": j["title"],
                    "Score": j["score"]
                }
                for j in ranked
            ])

            st.dataframe(df, use_container_width=True)

        # -------------------
        # GENERATED CONTENT
        # -------------------
        st.subheader("📝 Tailored Resume")
        st.write(result.get("tailored_resume", ""))

        st.subheader("📩 Cover Letter")
        st.write(result.get("cover_letter", ""))