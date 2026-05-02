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
        st.stop()

    with st.spinner("Analyzing resume with AI agent..."):
        result = run_graph(final_resume)

    st.success("Analysis complete!")

    # -------------------
    # TOP JOBS
    # -------------------
    st.subheader("🏆 Top Matching Jobs")

    ranked_jobs = sorted(
        result["scored_jobs"],
        key=lambda x: x["score"],
        reverse=True
    )

    for job in ranked_jobs[:5]:

        with st.container():
            st.markdown(f"### {job['title']}")
            st.write(job["description"])

            col1, col2 = st.columns([1, 3])

            with col1:
                st.metric("Score", job["score"])

            with col2:
                st.markdown(
                    f"[🚀 Apply Now]({job.get('apply_url', '#')})"
                )

            if job.get("details", {}).get("summary"):
                st.info(job["details"]["summary"])

            st.divider()

    # -------------------
    # RANKING TABLE
    # -------------------
    st.subheader("📊 Job Rankings")

    ranked = result.get("scored_jobs", [])

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