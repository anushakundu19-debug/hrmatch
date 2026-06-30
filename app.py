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
    prompt = f"""Extract resume info. Return JSON only.
Schema: {{"candidate_name":"","education":"","experience_years":0,"current_job_title":"","skills":[]}}
Resume: {text}"""
    result = call_llm(prompt)
    return parse_json_response(result)


def job_description_to_json(job_description: str) -> Dict:
    prompt = f"""Extract JD info. Return JSON only.
Schema: {{"job_title":"","education_required":"","required_skills":[],"responsibilities":[]}}
Job Description: {job_description}"""
    result = call_llm(prompt)
    return parse_json_response(result)


def analyze_resume_comprehensive(resume: Dict, jd: Dict) -> Dict:
    """
    COMBINED ANALYSIS: Compare resume, match skills, generate insights, training all in ONE call.
    Saves ~60% tokens by merging 4 separate analyses into 1.
    Only call recruiter insights if match_score >= 50%.
    """
    prompt = f"""You are an HR recruiter. Analyze this resume vs JD. Be concise.
Resume: {json.dumps(resume)}
JD: {json.dumps(jd)}

Return JSON only:
{{
  "match_score": 0,
  "matching_skills": [],
  "missing_skills": [],
  "synonym_matches": {{"candidate_skill":"jd_skill"}},
  "strengths": [],
  "areas_for_improvement": [],
  "hiring_recommendation": "hire/consider/review/pass"
}}"""
    result = call_llm(prompt)
    return parse_json_response(result)


def generate_training_recommendations(resume: Dict, jd: Dict, analysis: Dict) -> Dict:
    """
    Generate training with real certifications and courses (token-optimized).
    """
    if analysis.get('match_score', 0) < 30:
        return {
            "certifications": [],
            "recommended_courses": [],
            "project_recommendations": [],
            "learning_sequence": "Focus on foundational skills first",
            "estimated_total_timeline": "6-12 months",
            "quick_wins": ["Complete online overview courses", "Read industry whitepapers"]
        }
    
    prompt = f"""As career coach, suggest 2-3 certifications and 2-3 courses. Be concise.
Missing Skills: {json.dumps(analysis.get('missing_skills', [])[:5])}
Areas to Improve: {json.dumps(analysis.get('areas_for_improvement', [])[:3])}
Match Score: {analysis.get('match_score', 0)}%

Return JSON with max 3 items per list:
{{
  "certifications": [{{"certification_name":"", "issuing_body":"", "skill_addressed":"", "duration":"", "priority":"high/medium/low"}}],
  "recommended_courses": [{{"course_name":"", "platform":"", "skill":"", "duration":"", "difficulty_level":"beginner/intermediate/advanced", "cost":"free/paid"}}],
  "project_recommendations": [{{"project_type":"", "skills_gained":[], "complexity":"beginner/intermediate/advanced", "expected_duration":""}}],
  "learning_sequence": "order to follow",
  "estimated_total_timeline": "total months",
  "quick_wins": ["1-3 month achievements"]
}}"""
    result = call_llm(prompt)
    return parse_json_response(result)


def generate_recruiter_insights(resume: Dict, jd: Dict, analysis: Dict) -> Dict:
    """
    Only for high-potential candidates (match_score >= 50%).
    """
    match_score = analysis.get('match_score', 0)
    
    if match_score < 50:
        return {
            "synonym_insights": "See skill analysis above",
            "hidden_qualifications": analysis.get("strengths", []),
            "transferable_skills": [],
            "adjusted_match_score": match_score,
            "semantic_match_summary": f"Score: {match_score}%",
            "hiring_recommendation": analysis.get("hiring_recommendation", "review_manually"),
            "ramp_up_timeline": "6-12 months" if match_score >= 30 else "N/A - Major retraining needed",
            "risk_assessment": "Significant skill gaps. Strong training commitment required.",
            "interview_focus_areas": ["Why interested in this role?"],
            "additional_notes": ""
        }
    
    prompt = f"""Senior recruiter. Quick insights for match score {match_score}%.
Resume: {json.dumps(resume)}
Missing: {json.dumps(analysis.get('missing_skills', [])[:3])}
Strengths: {json.dumps(analysis.get('strengths', [])[:3])}

Return concise JSON:
{{
  "synonym_insights": "key terminology matches",
  "hidden_qualifications": ["implied skills"],
  "transferable_skills": ["other applicable skills"],
  "adjusted_match_score": {match_score},
  "semantic_match_summary": "brief summary",
  "hiring_recommendation": "hire/consider/review/pass",
  "ramp_up_timeline": "weeks/months",
  "risk_assessment": "brief risk summary",
  "interview_focus_areas": ["topic1", "topic2"],
  "additional_notes": ""
}}"""
    result = call_llm(prompt)
    return parse_json_response(result)


st.set_page_config(page_title="HRMatch", layout="wide")
st.title("HRMatch Resume Screening")
st.write("Upload PDFs and compare to job description. **Token-optimized for free tier (100K/day)** 🚀")

with st.sidebar:
    st.header("⚙️ Settings")
    st.info("💡 **Token Optimization:** Only generates advanced insights for strong candidates (50%+ match)")
    st.caption("Groq API key: Streamlit secrets → Shell env → .env file")

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select multiple files at once",
)
job_description = st.text_area("Job description", value=DEFAULT_JOB_DESCRIPTION, height=200)

if st.button("🔍 Analyze Resumes"):
    if not uploaded_files:
        st.error("Upload at least one PDF resume")
    elif not job_description.strip():
        st.error("Provide a job description")
    else:
        try:
            with st.spinner("Analyzing resumes (token-optimized)..."):
                jd = job_description_to_json(job_description)
                results: List[Dict] = []

                for uploaded_file in uploaded_files:
                    text = extract_text(uploaded_file.getvalue())
                    resume = resume_to_json(text)
                    
                    # OPTIMIZATION: Single comprehensive analysis call instead of 4-5 separate calls
                    analysis = analyze_resume_comprehensive(resume, jd)
                    
                    # Only generate recruiter insights for candidates with 50%+ match
                    if analysis.get("match_score", 0) >= 50:
                        recruiter_insights = generate_recruiter_insights(resume, jd, analysis)
                    else:
                        recruiter_insights = generate_recruiter_insights(resume, jd, analysis)
                    
                    # Generate training recommendations
                    training_recommendations = generate_training_recommendations(resume, jd, analysis)

                    results.append(
                        {
                            "candidate_name": resume.get("candidate_name", "Unknown"),
                            "current_job_title": resume.get("current_job_title", "Unknown"),
                            "experience_years": resume.get("experience_years", 0),
                            "education": resume.get("education", "Unknown"),
                            "match_score": analysis.get("match_score", 0),
                            "adjusted_match_score": recruiter_insights.get("adjusted_match_score", analysis.get("match_score", 0)),
                            "hiring_recommendation": recruiter_insights.get("hiring_recommendation", "review_manually"),
                            "matching_skills": analysis.get("matching_skills", []),
                            "missing_skills": analysis.get("missing_skills", []),
                            "synonym_matches": analysis.get("synonym_matches", {}),
                            "strengths": analysis.get("strengths", []),
                            "hidden_qualifications": recruiter_insights.get("hidden_qualifications", []),
                            "transferable_skills": recruiter_insights.get("transferable_skills", []),
                            "training_recommendations": training_recommendations,
                            "recruiter_insights": recruiter_insights,
                        }
                    )

                results_df = pd.DataFrame(results).sort_values(by="adjusted_match_score", ascending=False)

            st.success("✅ Analysis complete!")
            
            # Summary table
            display_df = results_df[["candidate_name", "match_score", "adjusted_match_score", "hiring_recommendation", "experience_years"]].copy()
            display_df.columns = ["Candidate", "Match %", "Adjusted %", "Recommendation", "Exp (yrs)"]
            st.dataframe(display_df, use_container_width=True)
            
            # Export
            export_df = results_df[["candidate_name", "current_job_title", "match_score", "adjusted_match_score", "hiring_recommendation"]].copy()
            st.download_button(
                label="📥 Download CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="hrmatch_results.csv",
                mime="text/csv",
            )

            # Detailed view
            st.header("📊 Detailed Results")
            for _, row in results_df.iterrows():
                with st.expander(f"**{row['candidate_name']}** — {row['adjusted_match_score']}% match"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 Overview")
                        st.write(f"**Title:** {row['current_job_title']}")
                        st.write(f"**Exp:** {row['experience_years']}y | **Edu:** {row['education']}")
                        st.write(f"**Score:** {row['match_score']}% → **{row['adjusted_match_score']}%**")
                    
                    with col2:
                        st.subheader("🎯 Recommendation")
                        st.write(f"**Action:** {row['hiring_recommendation']}")
                        st.write(f"**Timeline:** {row['recruiter_insights'].get('ramp_up_timeline', 'N/A')}")
                    
                    # Skills
                    st.subheader("🔧 Skills")
                    st.write("**Matching:** " + (", ".join(row["matching_skills"][:5]) or "None"))
                    st.write("**Synonyms:** " + (", ".join([f"{k}→{v}" for k,v in list(row["synonym_matches"].items())[:3]]) or "None"))
                    st.write("**Missing:** " + (", ".join(row["missing_skills"][:5]) or "None"))
                    
                    # Training
                    if row['training_recommendations'].get('certifications'):
                        st.subheader("🏆 Top Certifications")
                        for cert in row['training_recommendations']['certifications'][:2]:
                            st.write(f"• **{cert.get('certification_name')}** - {cert.get('duration')}")
                    
                    if row['training_recommendations'].get('recommended_courses'):
                        st.subheader("📚 Top Courses")
                        for course in row['training_recommendations']['recommended_courses'][:2]:
                            st.write(f"• **{course.get('course_name')}** ({course.get('platform')}) - {course.get('cost')}")
                    
                    st.write(f"⏱️ **Timeline:** {row['training_recommendations'].get('estimated_total_timeline')}")

        except Exception as exc:
            st.error(f"❌ Analysis failed: {exc}")
