import json
import os
import re
import time
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
    try:
        secret_value = st.secrets.get("GROQ_API_KEY")
    except Exception:
        secret_value = None

    api_key = (
        secret_value
        or os.getenv("GROQ_API_KEY")
        or st.session_state.get("groq_api_key", "")
    )
    if not api_key:
        raise ValueError(
            "Missing GROQ_API_KEY. Set it in Streamlit secrets, your shell, or a .env file."
        )
    return Groq(api_key=api_key)


def call_llm(prompt: str, model: str = "llama-3.3-70b-versatile", max_retries: int = 3) -> str:
    client = get_client()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            # Check for rate limit error (429)
            if "429" in error_str or "Rate limit" in error_str:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    st.warning(f"⏳ Rate limited. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error("❌ Rate limit exceeded. Please wait a few minutes and try again.")
                    raise
            else:
                raise

def normalize_match_score(score: float) -> int:
    """Normalize score to 0-100 range. If it's a decimal (0-1), multiply by 100."""
    if isinstance(score, str):
        try:
            score = float(score)
        except ValueError:
            return 0
    if score < 0:
        return 0
    if 0 <= score <= 1:
        return int(score * 100)
    if score > 100:
        return 100
    return int(score)


def parse_json_response(raw_response: str) -> Dict:
    if not raw_response:
        return {}

    cleaned = raw_response.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    if not cleaned:
        return {}

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    score_match = re.search(r"(?:match[_\s-]?score|score)\s*[:=]?\s*(\d+)", cleaned, flags=re.IGNORECASE)
    if score_match:
        return {"match_score": int(score_match.group(1))}

    key_value_pairs: Dict[str, object] = {}
    for line in cleaned.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().strip('"').strip("'")
        value = value.strip().strip('"').strip("'")
        try:
            key_value_pairs[key] = int(value)
        except ValueError:
            key_value_pairs[key] = value

    if key_value_pairs:
        return key_value_pairs

    raise ValueError("Model response was not valid JSON and could not be parsed.")


def resume_to_json(text: str) -> Dict:
    """Extract resume fields including company and industry exposure."""
    prompt = f"""Extract from resume: candidate_name, education, experience_years, current_job_title, current_company, previous_companies (list 2-3 most recent), skills (5 max). JSON only.
Resume: {text[:1000]}"""
    result = call_llm(prompt)
    return parse_json_response(result)


def job_description_to_json(job_description: str) -> Dict:
    """Extract JD fields. Ultra-minimal prompt."""
    prompt = f"""Extract: job_title, required_skills (5 max). JSON only.
JD: {job_description[:800]}"""
    result = call_llm(prompt)
    parsed = parse_json_response(result)
    if "job_title" in parsed:
        parsed["jd_job_title"] = parsed.pop("job_title")
    return parsed


def quick_match_analysis(resume: Dict, jd: Dict) -> Dict:
    """
    ULTRA-FAST screening. Returns only match_score.
    ~200 tokens. Determines if we do deeper analysis.
    """
    company_history = ""
    if resume.get("current_company"):
        company_history += f"Current: {resume.get('current_company')}. "
    if resume.get("previous_companies"):
        company_history += f"Previous: {', '.join(resume.get('previous_companies', []))}."
    
    prompt = f"""Score resume match on a scale of 0 to 100. Return ONLY JSON: {{"match_score": NUMBER}}

Scoring guide:
- 80-100: Direct match in role/skills. Ready to hire.
- 60-79: Good fit. Has most required skills or strong related experience. Trainable.
- 40-59: Moderate fit. Some relevant skills/experience. Would need training.
- 20-39: Weak fit. Different field but has transferable skills.
- 0-19: No relevant background.

Consider:
- Transferable skills (management, communication, problem-solving apply across roles)
- Years of experience in related domains
- Industry exposure from current/previous companies (tech/finance/HR/retail/manufacturing/etc)
- Soft skills like leadership, organization, people management
- Education and certifications

Resume Skills: {resume.get('skills', [])}
Resume Experience: {resume.get('experience_years', 0)} years as {resume.get('current_job_title')}
Company Background: {company_history}
Resume Education: {resume.get('education')}

Required Skills: {jd.get('required_skills', [])}
Target Role: {jd.get('jd_job_title')}"""
    result = call_llm(prompt)
    parsed = parse_json_response(result)
    if "match_score" in parsed:
        parsed["match_score"] = normalize_match_score(parsed["match_score"])
    return parsed


def comprehensive_analysis(resume: Dict, jd: Dict) -> Dict:
    """
    DETAILED analysis. Only for match_score >= 40%.
    Returns: match_score, matching_skills, missing_skills, strengths, synonyms.
    ~600 tokens.
    """
    company_history = ""
    if resume.get("current_company"):
        company_history += f"Currently at: {resume.get('current_company')}. "
    if resume.get("previous_companies"):
        company_history += f"Previously worked at: {', '.join(resume.get('previous_companies', []))}."
    
    prompt = f"""Analyze resume vs job description. Consider industry exposure from company background. Be fair in scoring. Return JSON with match_score (0-100).

Scoring Rubric:
- 80-100: Excellent fit. Has most/all required skills. Direct experience match. Strong industry alignment.
- 60-79: Strong fit. Has key skills. Some gaps but easily trainable. Related industry exposure.
- 40-59: Fair fit. Foundational skills present. Can learn on the job. Some relevant industry background.
- 20-39: Possible fit. Transferable skills from other domain.
- 0-19: Poor fit. No relevant background.

Consider both hard skills AND industry exposure from company history.

Resume Summary:
- Title: {resume.get('current_job_title')}
- Experience: {resume.get('experience_years')} years
- Company Background: {company_history}
- Education: {resume.get('education')}
- Skills: {resume.get('skills', [])}

Job Description:
- Role: {jd.get('jd_job_title')}
- Required Skills: {jd.get('required_skills', [])}

Return ONLY valid JSON:
{{"match_score": NUMBER, "matching_skills": ["list 3-5 matched skills"], "missing_skills": ["list 2-3 gaps"], "synonym_matches": {{"resume_skill":"matches_jd_skill"}}, "strengths": ["2-3 relevant strengths including industry expertise if applicable"]}}"""
    result = call_llm(prompt)
    parsed = parse_json_response(result)
    if "match_score" in parsed:
        parsed["match_score"] = normalize_match_score(parsed["match_score"])
    return parsed


def generate_minimal_training(missing_skills: List[str]) -> Dict:
    """
    Minimal training for <40% match. Just quick paths.
    ~300 tokens.
    """
    if not missing_skills:
        return {
            "certifications": [],
            "courses": [],
            "timeline": "Already qualified",
            "quick_wins": []
        }
    
    top_3_skills = missing_skills[:3]
    prompt = f"""Suggest 1 cert and 1 course for: {json.dumps(top_3_skills)}. Concise JSON.
{{"certifications": [{{"name":"", "issuing_body":"", "duration":""}}], "courses": [{{"name":"", "platform":"", "duration":""}}], "timeline": "3-6 months", "quick_wins": ["1-2 items"]}}"""
    result = call_llm(prompt)
    return parse_json_response(result)


def generate_detailed_training(resume: Dict, jd: Dict, analysis: Dict) -> Dict:
    """
    Detailed training for >= 40% match.
    ~700 tokens.
    """
    prompt = f"""Career coach. Suggest 2-3 certs and 2-3 courses. Concise JSON.
Missing skills: {json.dumps(analysis.get('missing_skills', [])[:3])}
Match: {analysis.get('match_score')}%

{{"certifications": [{{"name":"", "issuing_body":"", "duration":"", "priority":"high/med/low"}}], "courses": [{{"name":"", "platform":"", "duration":"", "cost":"free/paid"}}], "projects": [{{"type":"", "duration":""}}], "timeline": "X-Y months", "quick_wins": ["items"]}}"""
    result = call_llm(prompt)
    return parse_json_response(result)


def generate_recruiter_insights(analysis: Dict, match_score: int) -> Dict:
    """
    Only for >= 50% match. Returns hiring recommendation.
    ~300 tokens.
    """
    if match_score < 50:
        return {
            "hiring_recommendation": "needs_training" if match_score >= 40 else "pass",
            "ramp_up_timeline": "6-12 months" if match_score >= 40 else "N/A",
            "risk_level": "high" if match_score < 40 else "medium"
        }
    
    prompt = f"""Hiring decision for {match_score}% match. JSON: {{"hiring_recommendation":"hire/consider/review/pass", "ramp_up_timeline":"X months", "risk_level":"low/med/high"}}"""
    result = call_llm(prompt)
    return parse_json_response(result)


st.set_page_config(page_title="HRMatch", layout="wide")
st.title("HRMatch Resume Screening")
st.write("Fast token-optimized screening. **Ultra-efficient for 100K daily limit** ⚡")

with st.sidebar:
    st.header("⚙️ Settings")
    st.info("🚀 **Ultra-Optimized:**\n- Quick match scan first\n- Detailed analysis only for 40%+ matches\n- Minimal training for weak candidates\n- ~50 resumes/day on 100K limit")
    st.caption("Groq API key: Streamlit secrets → .env")

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
)
job_description = st.text_area("Job description", value=DEFAULT_JOB_DESCRIPTION, height=180)

if st.button("⚡ Quick Scan"):
    if not uploaded_files:
        st.error("Upload at least one PDF")
    elif not job_description.strip():
        st.error("Provide a job description")
    else:
        try:
            with st.spinner("Scanning resumes..."):
                jd = job_description_to_json(job_description)
                results: List[Dict] = []
                detailed_needed = []

                # PHASE 1: Quick scan all resumes
                for idx, uploaded_file in enumerate(uploaded_files):
                    text = extract_text(uploaded_file.getvalue())
                    resume = resume_to_json(text)
                    quick_score = quick_match_analysis(resume, jd)
                    
                    results.append({
                        "file_idx": idx,
                        "candidate_name": resume.get("candidate_name", "Unknown"),
                        "current_job_title": resume.get("current_job_title", "Unknown"),
                        "experience_years": resume.get("experience_years", 0),
                        "education": resume.get("education", "Unknown"),
                        "match_score": quick_score.get("match_score", 0),
                        "resume": resume,
                        "status": "quick_scan_only"
                    })
                    
                    # Mark for detailed analysis if promising
                    if quick_score.get("match_score", 0) >= 40:
                        detailed_needed.append(idx)

                # PHASE 2: Detailed analysis only for promising candidates
                st.info(f"📊 Scanned {len(results)} resumes. Analyzing {len(detailed_needed)} promising candidates...")
                
                for file_idx in detailed_needed:
                    resume = results[file_idx]["resume"]
                    analysis = comprehensive_analysis(resume, jd)
                    
                    # Only generate detailed training for 40%+ match
                    if analysis.get("match_score", 0) >= 40:
                        training = generate_detailed_training(resume, jd, analysis)
                    else:
                        training = generate_minimal_training(analysis.get("missing_skills", []))
                    
                    # Only generate recruiter insights for 50%+ match
                    recruiter_insights = generate_recruiter_insights(analysis, analysis.get("match_score", 0))
                    
                    results[file_idx].update({
                        "comprehensive_score": analysis.get("match_score", 0),
                        "matching_skills": analysis.get("matching_skills", []),
                        "missing_skills": analysis.get("missing_skills", []),
                        "synonym_matches": analysis.get("synonym_matches", {}),
                        "strengths": analysis.get("strengths", []),
                        "training": training,
                        "recruiter_insights": recruiter_insights,
                        "status": "detailed_analysis"
                    })

                results_df = pd.DataFrame(results)
                results_df["ranking_score"] = results_df.apply(
                    lambda row: row.get("comprehensive_score", row["match_score"]) 
                    if row["status"] == "detailed_analysis" 
                    else row["match_score"],
                    axis=1
                )
                results_df = results_df.sort_values(by="ranking_score", ascending=False)

            st.success("✅ Analysis complete!")
            
            # Summary
            st.subheader("📋 Quick Results")
            display_df = results_df[["candidate_name", "match_score", "current_job_title", "experience_years"]].copy()
            display_df.columns = ["Candidate", "Match %", "Title", "Exp (yrs)"]
            
            # Show comprehensive score if detailed analysis was done
            display_df["Match %"] = display_df.index.map(
                lambda idx: results_df.loc[idx, "comprehensive_score"] 
                if results_df.loc[idx, "status"] == "detailed_analysis" 
                else results_df.loc[idx, "match_score"]
            )
            display_df["Match %"] = display_df["Match %"].apply(lambda x: f"{int(x)}%")
            st.dataframe(display_df, use_container_width=True)
            
            # Export
            export_df = results_df[["candidate_name", "current_job_title", "match_score", "experience_years"]].copy()
            st.download_button(
                label="📥 Download CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="hrmatch_results.csv",
                mime="text/csv",
            )

            # Detailed view
            st.header("📊 Candidates")
            for _, row in results_df.iterrows():
                quick_match = row["match_score"]
                comprehensive_match = row.get("comprehensive_score", quick_match)
                display_match = comprehensive_match if row['status'] == 'detailed_analysis' else quick_match
                status_emoji = "🟢" if display_match >= 70 else "🟡" if display_match >= 40 else "🔴"
                
                with st.expander(f"{status_emoji} **{row['candidate_name']}** — {int(display_match)}% match"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Title:** {row['current_job_title']}")
                        st.write(f"**Exp:** {row['experience_years']}y")
                    
                    with col2:
                        if row['status'] == 'detailed_analysis':
                            st.write(f"**Quick Score:** {quick_match}%")
                            st.write(f"**Detailed Score:** {comprehensive_match}%")
                        else:
                            st.write(f"**Match:** {quick_match}%")
                        st.write(f"**Edu:** {row['education']}")
                    
                    with col3:
                        if row['status'] == 'detailed_analysis':
                            rec = row.get('recruiter_insights', {}).get('hiring_recommendation', 'review')
                            st.write(f"**Action:** {rec}")
                            st.write(f"**Risk:** {row.get('recruiter_insights', {}).get('risk_level', 'unknown')}")
                        else:
                            st.write("⚡ Quick scan")
                    
                    # Show detailed results only if analyzed
                    if row['status'] == 'detailed_analysis':
                        st.divider()
                        
                        col_skills1, col_skills2 = st.columns(2)
                        with col_skills1:
                            st.write("**✅ Matching:**")
                            for skill in row.get("matching_skills", [])[:3]:
                                st.write(f"- {skill}")
                        
                        with col_skills2:
                            st.write("**❌ Missing:**")
                            for skill in row.get("missing_skills", [])[:3]:
                                st.write(f"- {skill}")
                        
                        # Training recommendations
                        if row.get('training'):
                            st.divider()
                            training = row['training']
                            
                            if training.get('certifications'):
                                st.write("**🏆 Cert:** " + (training['certifications'][0].get('name', 'N/A') if training['certifications'] else "N/A"))
                            
                            if training.get('courses'):
                                st.write("**📚 Course:** " + (training['courses'][0].get('name', 'N/A') if training['courses'] else "N/A"))
                            
                            st.write(f"⏱️ **Timeline:** {training.get('timeline', 'N/A')}")

        except Exception as exc:
            error_msg = str(exc)
            if "429" in error_msg or "Rate limit" in error_msg:
                st.error(
                    "❌ **Rate limit reached.** The Groq API has temporarily limited requests. "
                    "Please wait a few minutes and try again with fewer resumes at once."
                )
            else:
                st.error(f"❌ Error: {error_msg[:150]}")
