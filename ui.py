import streamlit as st
from engine import run_graph, extract_text_from_pdf

st.title("Career Copilot AI")

st.write("Upload your resume (PDF) or paste text")

# -------------------
# INPUT OPTION 1: PDF
# -------------------
uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

# -------------------
# INPUT OPTION 2: TEXT
# -------------------
resume_text = st.text_area("Or Paste Resume Text")

final_resume = None

# Decide input source
if uploaded_file is not None:
    final_resume = extract_text_from_pdf(uploaded_file)
    st.success("PDF loaded successfully")

elif resume_text:
    final_resume = resume_text

# -------------------
# RUN BUTTON
# -------------------
if st.button("Run AI Career Agent"):
    if not final_resume:
        st.error("Please upload a PDF or paste resume text")
    else:
        with st.spinner("Analyzing resume..."):
            result = run_graph(final_resume)

        st.success("Done!")

        st.subheader("Best Job Match")
        st.write(result.get("best_job", {}))

        st.subheader("Tailored Resume")
        st.write(result.get("tailored_resume", ""))

        st.subheader("Cover Letter")
        st.write(result.get("cover_letter", ""))