import streamlit as st
from engine import run_graph

st.title("Career Copilot AI")

resume = st.text_area("Paste Resume")

if st.button("Run AI Agent"):
    if resume:
        result = run_graph(resume)

        st.subheader("Best Job")
        st.write(result["best_job"])

        st.subheader("Resume")
        st.write(result["tailored_resume"])

        st.subheader("Cover Letter")
        st.write(result["cover_letter"])