import json
import os
from typing import Dict, List

import fitz
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_JOB_DESCRIPTION = """
HR Business Partner Job Description

Responsibilities
- Collaborate with senior leadership to align HR strategies with business objectives.
- Support succession planning, career development, and workforce planning.
- Implement performance review systems and coach managers.
- Resolve employee relations issues and improve engagement.
- Guide leaders through organizational change and restructures.
- Ensure compliance with HR policies and labor regulations.
- Use HR analytics to support data-driven decision making.

Qualifications
- Any Graduate
- Strong stakeholder management and communication skills.
- Experience in HR operations, employee relations, and talent management.
"""


def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def get_client() -> Groq:
    api_key = (
        st.secrets.get("GROQ_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or st.session_state.get("groq_api_key", "")
    )
    if not api_key:
        raise ValueError(
            "Missing GROQ_API_KEY. Set it in Streamlit secrets, your shell, or a .env file."
        )
    return Groq(api_key=api_key)


def call_llm(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def parse_json_response(raw_response: str) -> Dict:
    cleaned = raw_response.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def resume_to_json(text: str) -> Dict:
    prompt = f"""
You are an HR recruiter.

Extract information from the resume below and return ONLY valid JSON.

Schema:
{{
  "candidate_name": "",
  "education": "",
  "experience_years": 0,
  "current_job_title": "",
  "skills": []
}}

Resume:
{text}
"""
    result = call_llm(prompt)
    return parse_json_response(result)


def job_description_to_json(job_description: str) -> Dict:
    prompt = f"""
You are an HR recruiter.

Read the job description below and return ONLY valid JSON.

Schema:
{{
  "job_title": "",
  "education_required": "",
  "required_skills": [],
  "responsibilities": []
}}

Job Description:
{job_description}
"""
    result = call_llm(prompt)
    return parse_json_response(result)


def compare_resume_to_jd(resume: Dict, jd: Dict) -> Dict:
    prompt = f"""
You are a Senior HR Recruiter.

Candidate Resume:
{json.dumps(resume, indent=2)}

Job Description:
{json.dumps(jd, indent=2)}

Compare the candidate against the job description and return ONLY valid JSON.

Schema:
{{
  "match_score": 0,
  "matching_skills": [],
  "missing_skills": [],
  "strengths": [],
  "areas_for_improvement": [],
  "recommendation": ""
}}
"""
    result = call_llm(prompt)
    return parse_json_response(result)


st.set_page_config(page_title="HRMatch", layout="wide")
st.title("HRMatch Resume Screening")
st.write("Upload multiple PDF resumes at once and compare them to a job description using Groq.")

with st.sidebar:
    st.header("Settings")
    st.caption("Add your Groq API key before running the analysis.")
    st.code("export GROQ_API_KEY=your_key_here", language="bash")
    st.code("cp .env.example .env\n# then edit .env and add your key", language="bash")

    if "groq_api_key" not in st.session_state:
        st.session_state["groq_api_key"] = os.getenv("GROQ_API_KEY", "")

    api_key_input = st.text_input(
        "Groq API key",
        value=st.session_state["groq_api_key"],
        type="password",
        key="groq_api_key",
        help="The app will reuse the saved key automatically for this session.",
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input
        st.session_state["groq_api_key"] = api_key_input

    st.info("The app reads GROQ_API_KEY from the textbox, your shell environment, or a .env file in this workspace.")

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can select several resume files in one upload dialog.",
)
job_description = st.text_area("Job description", value=DEFAULT_JOB_DESCRIPTION, height=250)

if st.button("Analyze Resumes"):
    if not uploaded_files:
        st.error("Please upload at least one PDF resume.")
    elif not job_description.strip():
        st.error("Please provide a job description.")
    else:
        try:
            with st.spinner("Analyzing resumes..."):
                jd = job_description_to_json(job_description)
                results: List[Dict] = []

                for uploaded_file in uploaded_files:
                    text = extract_text(uploaded_file.getvalue())
                    resume = resume_to_json(text)
                    comparison = compare_resume_to_jd(resume, jd)

                    results.append(
                        {
                            "candidate_name": resume.get("candidate_name", "Unknown"),
                            "current_job_title": resume.get("current_job_title", "Unknown"),
                            "experience_years": resume.get("experience_years", 0),
                            "education": resume.get("education", "Unknown"),
                            "match_score": comparison.get("match_score", 0),
                            "recommendation": comparison.get("recommendation", "Review manually"),
                            "matching_skills": comparison.get("matching_skills", []),
                            "missing_skills": comparison.get("missing_skills", []),
                        }
                    )

                results_df = pd.DataFrame(results).sort_values(by="match_score", ascending=False)

            st.success("Analysis complete.")
            export_df = results_df.copy()
            export_df["matching_skills"] = export_df["matching_skills"].apply(lambda skills: json.dumps(skills))
            export_df["missing_skills"] = export_df["missing_skills"].apply(lambda skills: json.dumps(skills))
            st.dataframe(results_df, use_container_width=True)
            st.download_button(
                label="Download results as CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="resume_screening_results.csv",
                mime="text/csv",
            )

            for _, row in results_df.iterrows():
                with st.expander(f"{row['candidate_name']} — {row['match_score']}% match"):
                    st.write(f"**Current Job Title:** {row['current_job_title']}")
                    st.write(f"**Education:** {row['education']}")
                    st.write(f"**Experience:** {row['experience_years']} years")
                    st.write(f"**Recommendation:** {row['recommendation']}")
                    st.write("**Matching Skills:**")
                    for skill in row["matching_skills"]:
                        st.write(f"- {skill}")
                    st.write("**Missing Skills:**")
                    for skill in row["missing_skills"]:
                        st.write(f"- {skill}")

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
