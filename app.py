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


def normalize_skills_with_synonyms(candidate_skills: List[str], jd_skills: List[str]) -> Dict:
    """
    Perform semantic matching to identify synonyms and equivalent terms.
    Returns mapping of candidate skills to matched JD skills.
    """
    prompt = f"""
You are an expert in skill taxonomy and job market semantics.

Analyze the candidate's skills and job description requirements. 
Look for synonyms, equivalent terms, and skills with different naming conventions 
(e.g., "Python" vs "Py", "Machine Learning" vs "ML", "REST API" vs "RESTful API", etc.).

Candidate Skills:
{json.dumps(candidate_skills, indent=2)}

Job Description Required Skills:
{json.dumps(jd_skills, indent=2)}

Return ONLY valid JSON with:
1. Direct matches (exact or obvious equivalents)
2. Synonym matches (skills that mean the same thing but are named differently)
3. Unmatched candidate skills
4. Unmatched JD skills

Schema:
{{
  "direct_matches": [],
  "synonym_matches": {{"candidate_skill": "equivalent_jd_skill"}},
  "unmatched_candidate_skills": [],
  "unmatched_jd_skills": []
}}
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


def generate_training_recommendations(resume: Dict, jd: Dict, comparison: Dict) -> Dict:
    """
    Generate personalized training and project recommendations for the candidate
    with specific certification names and course recommendations.
    """
    prompt = f"""
You are a Career Development Coach specializing in personalized learning paths and industry certifications.

Candidate Profile:
{json.dumps(resume, indent=2)}

Target Job Requirements:
{json.dumps(jd, indent=2)}

Gap Analysis:
- Missing Skills: {json.dumps(comparison.get('missing_skills', []))}
- Areas for Improvement: {json.dumps(comparison.get('areas_for_improvement', []))}
- Current Match Score: {comparison.get('match_score', 0)}%

Based on the gaps identified, create a personalized development plan with REAL, SPECIFIC certifications and courses.
Include actual certification names, recognized credentials, and well-known platforms.

For certifications, include recognized credentials like:
- HR certifications: SHRM-CP, SHRM-SCP, PHR, CIPD, IHRP
- Project Management: PMP, CAPM, PRINCE2
- Analytics: Google Analytics, Microsoft Certifications, Tableau Desktop Specialist
- Leadership: CCP (Center for Creative Leadership), EY Future Consumer Index Certification
- Compliance: GDPR certification, SOC 2 fundamentals
- Technical: AWS, Azure, Google Cloud certifications (if applicable)

For courses, recommend platforms like: Coursera, Udemy, LinkedIn Learning, edX, SHRM Learning, HR.com, Pluralsight

Return ONLY valid JSON with SPECIFIC, REAL certifications and course names:

Schema:
{{
  "certifications": [
    {{
      "certification_name": "Official certification name (e.g., 'SHRM-CP Certification')",
      "issuing_body": "Organization issuing the cert (e.g., 'Society for Human Resource Management')",
      "skill_addressed": "which skill this certifies",
      "duration": "how long typically takes to complete",
      "cost_estimate": "estimated cost range",
      "priority": "high/medium/low",
      "why_important": "why this specific certification helps the role"
    }}
  ],
  "recommended_courses": [
    {{
      "course_name": "Exact course name from platform",
      "platform": "Coursera/Udemy/LinkedIn Learning/etc",
      "skill": "skill addressed",
      "duration": "estimated duration",
      "difficulty_level": "beginner/intermediate/advanced",
      "cost": "free/paid/subscription",
      "estimated_price": "price range if paid",
      "why_relevant": "relevance to the job"
    }}
  ],
  "project_recommendations": [
    {{
      "project_type": "project description",
      "skills_gained": ["skill1", "skill2"],
      "complexity": "beginner/intermediate/advanced",
      "expected_duration": "estimated time to complete",
      "business_value": "how this helps the role"
    }}
  ],
  "learning_sequence": "suggested order to tackle certifications and courses",
  "estimated_total_timeline": "approximate total time to reach proficiency",
  "quick_wins": ["achievable certifications/skills in 1-3 months"],
  "budget_estimate": "estimated total cost for all recommendations"
}}
"""
    result = call_llm(prompt)
    return parse_json_response(result)


def generate_recruiter_insights(resume: Dict, jd: Dict, comparison: Dict, synonym_analysis: Dict) -> Dict:
    """
    Generate deeper insights for HR recruiters, including semantic skill matching
    and hidden qualifications.
    """
    prompt = f"""
You are a Senior HR Recruiter and Talent Acquisition Specialist.

You have performed both exact and semantic skill matching. 
The synonym analysis has already identified skills with different naming conventions.

Candidate Resume:
{json.dumps(resume, indent=2)}

Job Description:
{json.dumps(jd, indent=2)}

Synonym Matching Results:
{json.dumps(synonym_analysis, indent=2)}

Initial Comparison:
{json.dumps(comparison, indent=2)}

Provide deeper insights for recruiting teams, considering:
1. Hidden qualifications (implied skills from job history)
2. Transferable skills the candidate might not have listed
3. Synonym matches and equivalent terminology
4. Risk assessment for skill gaps
5. Ramp-up timeline estimate

Return ONLY valid JSON:

Schema:
{{
  "synonym_insights": "summary of skills that match using different terminology",
  "hidden_qualifications": ["implied skills from candidate's background"],
  "transferable_skills": ["skills from other roles that apply here"],
  "adjusted_match_score": "recalculated score after semantic matching (0-100)",
  "semantic_match_summary": "explanation of semantic matches found",
  "hiring_recommendation": "hire/consider/review_manually/pass",
  "ramp_up_timeline": "how long to reach full productivity",
  "risk_assessment": "potential gaps and mitigation strategies",
  "interview_focus_areas": ["topics to explore in interviews"],
  "additional_notes": ""
}}
"""
    result = call_llm(prompt)
    return parse_json_response(result)


st.set_page_config(page_title="HRMatch", layout="wide")
st.title("HRMatch Resume Screening")
st.write("Upload multiple PDF resumes at once and compare them to a job description using Groq with AI-powered insights.")

with st.sidebar:
    st.header("Settings")
    st.caption("Groq API key is read from Streamlit secrets for this deployment.")
    st.code("# To run locally, create a .env file with:\n# GROQ_API_KEY=your_real_key_here", language="bash")
    st.info("If you deployed on Streamlit Cloud, set GROQ_API_KEY in App Settings → Secrets.")

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
                    
                    # Enhanced analysis: semantic skill matching
                    candidate_skills = resume.get("skills", [])
                    jd_skills = jd.get("required_skills", [])
                    synonym_analysis = normalize_skills_with_synonyms(candidate_skills, jd_skills)
                    
                    # Generate insights for HR recruiters
                    recruiter_insights = generate_recruiter_insights(resume, jd, comparison, synonym_analysis)
                    
                    # Generate recommendations for candidates (now with specific certs)
                    training_recommendations = generate_training_recommendations(resume, jd, comparison)

                    results.append(
                        {
                            "candidate_name": resume.get("candidate_name", "Unknown"),
                            "current_job_title": resume.get("current_job_title", "Unknown"),
                            "experience_years": resume.get("experience_years", 0),
                            "education": resume.get("education", "Unknown"),
                            "match_score": comparison.get("match_score", 0),
                            "adjusted_match_score": recruiter_insights.get("adjusted_match_score", comparison.get("match_score", 0)),
                            "recommendation": comparison.get("recommendation", "Review manually"),
                            "hiring_recommendation": recruiter_insights.get("hiring_recommendation", ""),
                            "matching_skills": comparison.get("matching_skills", []),
                            "missing_skills": comparison.get("missing_skills", []),
                            "synonym_matches": synonym_analysis.get("synonym_matches", {}),
                            "hidden_qualifications": recruiter_insights.get("hidden_qualifications", []),
                            "transferable_skills": recruiter_insights.get("transferable_skills", []),
                            "training_recommendations": training_recommendations,
                            "recruiter_insights": recruiter_insights,
                            "comparison": comparison,
                            "synonym_analysis": synonym_analysis,
                        }
                    )

                results_df = pd.DataFrame(results).sort_values(by="adjusted_match_score", ascending=False)

            st.success("Analysis complete.")
            
            # Display summary table with both original and adjusted scores
            display_df = results_df[["candidate_name", "match_score", "adjusted_match_score", "hiring_recommendation", "experience_years"]].copy()
            display_df.columns = ["Candidate", "Match Score %", "Adjusted Match %", "Hiring Recommendation", "Experience (years)"]
            st.dataframe(display_df, use_container_width=True)
            
            # Export functionality
            export_df = results_df[["candidate_name", "current_job_title", "experience_years", "education", 
                                    "match_score", "adjusted_match_score", "recommendation", "hiring_recommendation"]].copy()
            export_df["matching_skills"] = results_df["matching_skills"].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
            export_df["missing_skills"] = results_df["missing_skills"].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
            export_df["synonym_matches"] = results_df["synonym_matches"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
            
            st.download_button(
                label="Download results as CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="resume_screening_results.csv",
                mime="text/csv",
            )

            # Detailed view for each candidate
            st.header("Detailed Analysis")
            for idx, (_, row) in enumerate(results_df.iterrows()):
                with st.expander(f"{row['candidate_name']} — {row['adjusted_match_score']}% match (Adjusted)", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 Candidate Overview")
                        st.write(f"**Current Job Title:** {row['current_job_title']}")
                        st.write(f"**Education:** {row['education']}")
                        st.write(f"**Experience:** {row['experience_years']} years")
                        st.write(f"**Match Score:** {row['match_score']}% → **{row['adjusted_match_score']}%** (after semantic analysis)")
                    
                    with col2:
                        st.subheader("🎯 HR Recruiter Insights")
                        st.write(f"**Hiring Recommendation:** {row['hiring_recommendation']}")
                        st.write(f"**Ramp-up Timeline:** {row['recruiter_insights'].get('ramp_up_timeline', 'N/A')}")
                        st.info(f"**Semantic Match Summary:** {row['recruiter_insights'].get('semantic_match_summary', 'N/A')}")
                    
                    st.divider()
                    
                    # Skills Analysis Tab
                    st.subheader("🔧 Skills Analysis")
                    skills_col1, skills_col2, skills_col3 = st.columns(3)
                    
                    with skills_col1:
                        st.write("**✅ Matching Skills:**")
                        for skill in row["matching_skills"]:
                            st.write(f"- {skill}")
                    
                    with skills_col2:
                        st.write("**⚡ Synonym Matches:**")
                        for candidate_skill, jd_skill in row["synonym_matches"].items():
                            st.write(f"- {candidate_skill} → {jd_skill}")
                    
                    with skills_col3:
                        st.write("**❌ Missing Skills:**")
                        for skill in row["missing_skills"]:
                            st.write(f"- {skill}")
                    
                    st.divider()
                    
                    # Hidden Qualifications & Transferable Skills
                    st.subheader("🔍 Hidden Qualifications & Transferable Skills")
                    hidden_col, transfer_col = st.columns(2)
                    
                    with hidden_col:
                        st.write("**Hidden Qualifications (Implied from background):**")
                        for qual in row["hidden_qualifications"]:
                            st.write(f"• {qual}")
                    
                    with transfer_col:
                        st.write("**Transferable Skills:**")
                        for skill in row["transferable_skills"]:
                            st.write(f"• {skill}")
                    
                    st.divider()
                    
                    # Recommendation for Candidates
                    st.subheader("🎓 Personalized Training & Development Plan")
                    training_recs = row["training_recommendations"]
                    
                    st.write(f"**Estimated Total Timeline:** {training_recs.get('estimated_total_timeline', 'N/A')}")
                    st.write(f"**Learning Sequence:** {training_recs.get('learning_sequence', 'N/A')}")
                    st.write(f"**Budget Estimate:** {training_recs.get('budget_estimate', 'N/A')}")
                    
                    st.write("**Quick Wins (Achievable in 1-3 months):**")
                    for win in training_recs.get("quick_wins", []):
                        st.write(f"✓ {win}")
                    
                    # Certifications Section
                    st.write("**🏆 Recommended Certifications:**")
                    for cert in training_recs.get("certifications", []):
                        with st.container():
                            st.write(f"**{cert.get('certification_name', 'N/A')}**")
                            st.write(f"  - Issuing Body: {cert.get('issuing_body', 'N/A')}")
                            st.write(f"  - Skill Addressed: {cert.get('skill_addressed', 'N/A')}")
                            st.write(f"  - Duration: {cert.get('duration', 'N/A')}")
                            st.write(f"  - Estimated Cost: {cert.get('cost_estimate', 'N/A')}")
                            st.write(f"  - Priority: {cert.get('priority', 'N/A')}")
                            st.write(f"  - Why Important: {cert.get('why_important', 'N/A')}")
                    
                    # Recommended Courses Section
                    st.write("**📚 Recommended Courses:**")
                    for course in training_recs.get("recommended_courses", []):
                        with st.container():
                            st.write(f"**{course.get('course_name', 'N/A')}**")
                            st.write(f"  - Platform: {course.get('platform', 'N/A')}")
                            st.write(f"  - Skill: {course.get('skill', 'N/A')}")
                            st.write(f"  - Duration: {course.get('duration', 'N/A')}")
                            st.write(f"  - Level: {course.get('difficulty_level', 'N/A')}")
                            st.write(f"  - Cost: {course.get('cost', 'N/A')} {course.get('estimated_price', '')}")
                            st.write(f"  - Why Relevant: {course.get('why_relevant', 'N/A')}")
                    
                    st.write("**Recommended Projects:**")
                    for project in training_recs.get("project_recommendations", []):
                        with st.container():
                            st.write(f"📁 **{project.get('project_type', 'N/A')}**")
                            st.write(f"  - Skills Gained: {', '.join(project.get('skills_gained', []))}")
                            st.write(f"  - Complexity: {project.get('complexity', 'N/A')}")
                            st.write(f"  - Duration: {project.get('expected_duration', 'N/A')}")
                            st.write(f"  - Business Value: {project.get('business_value', 'N/A')}")
                    
                    st.divider()
                    
                    # HR Recruiter Deep Dive
                    st.subheader("📊 HR Recruiter Deep Dive")
                    st.write(f"**Interview Focus Areas:**")
                    for area in row["recruiter_insights"].get("interview_focus_areas", []):
                        st.write(f"• {area}")
                    
                    st.write(f"**Risk Assessment & Mitigation:**")
                    st.write(row["recruiter_insights"].get("risk_assessment", "N/A"))
                    
                    if row["recruiter_insights"].get("additional_notes"):
                        st.info(f"**Additional Notes:** {row['recruiter_insights'].get('additional_notes')}")

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
