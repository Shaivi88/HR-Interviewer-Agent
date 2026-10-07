import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from pipeline.graph import build_graph
from pipeline.state import ResumeEntry
from utils.pdf import extract_text

st.set_page_config(page_title="Job Application Screener", layout="wide")
st.title("Job Application Screener")

jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
resume_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

run_disabled = not jd_file or not resume_files
if st.button("Run Pipeline", disabled=run_disabled):
    try:
        jd_text = extract_text(jd_file.read())
    except ValueError as e:
        st.error(f"Job Description: {e}")
        st.stop()

    resumes: list[ResumeEntry] = []
    for f in resume_files:
        try:
            text = extract_text(f.read())
            resumes.append({
                "filename": f.name,
                "text": text,
                "score": 0,
                "reasoning": "",
                "questions": [],
            })
        except ValueError as e:
            st.error(f"{f.name}: {e}")
            st.stop()

    with st.spinner("Analyzing candidates..."):
        graph = build_graph()
        result = graph.invoke({"jd_text": jd_text, "resumes": resumes})

    ranked = result["resumes"]

    st.subheader("Results")
    table_data = [
        {
            "Rank": i + 1,
            "File": r["filename"],
            "Score": r["score"],
            "Reasoning": r["reasoning"],
        }
        for i, r in enumerate(ranked)
    ]
    st.dataframe(table_data, use_container_width=True)

    st.subheader("Interview Questions")
    for i, candidate in enumerate(ranked):
        with st.expander(
            f"#{i+1} — {candidate['filename']}  (Score: {candidate['score']})",
            expanded=(i == 0),
        ):
            for j, q in enumerate(candidate["questions"], 1):
                st.markdown(f"**{j}.** {q}")
