import io
import json
from datetime import datetime

import streamlit as st

from utils.model_loader import get_spacy_nlp, get_sentence_model
from utils.resume_parser import parse_resume, extract_sections, extract_contact_info, extract_skills_from_text
from utils.skill_matching import (
    compute_semantic_similarity,
    extract_keywords_from_text,
    compute_overlap_score,
    build_match_breakdown,
)
from utils.ats_checker import ats_check

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🧠",
    layout="wide",
)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Settings")
    model_name = st.selectbox(
        "Embedding model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ],
        index=0,
        help="Smaller models are faster; mpnet is more accurate but heavier."
    )
    w_sem = st.slider("Weight: Semantic similarity", 0.0, 1.0, 0.6, 0.05)
    w_kw = 1.0 - w_sem
    st.caption(f"Keyword/skill weight is automatically set to **{w_kw:.2f}**")
    min_skill_len = st.slider("Min skill length (chars)", 2, 10, 3, 1)
    show_raw = st.checkbox("Show raw parsed text", value=False)

st.title("🧠 AI Resume Matcher")
st.write("Upload your resume and paste a job description to get a match score, missing skills, and ATS suggestions.")

# ---------- Inputs ----------
col1, col2 = st.columns([1, 1])
with col1:
    resume_file = st.file_uploader(
        "Upload Resume (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        help="A clean, text-based resume works best."
    )
with col2:
    job_desc = st.text_area(
        "Paste Job Description",
        height=300,
        placeholder="Paste the job description here..."
    )

run_btn = st.button("🚀 Analyze Match", use_container_width=True)

# ---------- Model load (lazy) ----------
@st.cache_resource(show_spinner=False)
def _load_models(model_name_):
    nlp = get_spacy_nlp()
    sbert = get_sentence_model(model_name_)
    return nlp, sbert

if run_btn:
    if not resume_file or not job_desc.strip():
        st.warning("Please upload a resume and paste a job description.")
        st.stop()

    with st.spinner("Parsing resume..."):
        parsed = parse_resume(resume_file)

    if not parsed.text.strip():
        st.error("Couldn't extract text from the resume. Try a simpler PDF/DOCX or TXT.")
        st.stop()

    if show_raw:
        with st.expander("📄 Raw Parsed Resume Text"):
            st.write(parsed.text)

    nlp, sbert = _load_models(model_name)

    # ---------- NLP Extractions ----------
    with st.spinner("Extracting sections & contact details..."):
        sections = extract_sections(parsed.text)
        contact = extract_contact_info(parsed.text)
        resume_skills = extract_skills_from_text(parsed.text, nlp=nlp, min_len=min_skill_len)
        jd_keywords = extract_keywords_from_text(job_desc, nlp=nlp, min_len=min_skill_len)

    # ---------- Similarity ----------
    with st.spinner("Computing similarity & overlap..."):
        sim_score = compute_semantic_similarity(parsed.text, job_desc, model=sbert)
        overlap = compute_overlap_score(resume_skills, jd_keywords)
        final_score, breakdown = build_match_breakdown(
            sim_score=sim_score,
            overlap_score=overlap,
            w_sem=w_sem,
            w_kw=w_kw
        )

    # ---------- ATS Check ----------
    with st.spinner("Running ATS checks..."):
        ats_report = ats_check(
            text=parsed.text,
            filetype=parsed.filetype,
            page_count=parsed.page_count,
            sections=sections,
            extracted_skills=resume_skills,
            jd_keywords=jd_keywords
        )

    # ---------- UI: Scores ----------
    st.subheader("📊 Match Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Match", f"{final_score:.1f} / 100")
    c2.metric("Semantic Similarity", f"{sim_score*100:.1f} / 100")
    c3.metric("Keyword/Skill Overlap", f"{overlap*100:.1f} / 100")

    with st.expander("🔎 Score Breakdown"):
        st.json(breakdown, expanded=False)

    # ---------- Missing Skills ----------
    st.subheader("🧩 Skills & Keywords")
    missing = [k for k in jd_keywords if k not in resume_skills]
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Detected Resume Skills:**")
        if resume_skills:
            st.write(", ".join(sorted(set(resume_skills))))
        else:
            st.write("_No skills detected — consider a Skills section with comma-separated items._")
    with col_b:
        st.markdown("**JD Keywords (Top):**")
        st.write(", ".join(sorted(set(jd_keywords))))
    st.markdown("**Recommended to Add / Emphasize:**")
    if missing:
        st.info(", ".join(sorted(set(missing))))
    else:
        st.success("Great! Your resume covers most of the JD keywords.")

    # ---------- Sections & Contact ----------
    st.subheader("🧭 Resume Structure")
    s1, s2 = st.columns([2, 1])
    with s1:
        for sec, txt in sections.items():
            if txt.strip():
                with st.expander(f"**{sec}**"):
                    st.write(txt.strip()[:2000] + ("..." if len(txt) > 2000 else ""))
    with s2:
        st.markdown("**Contact Details (detected):**")
        st.json(contact, expanded=False)

    # ---------- ATS Suggestions ----------
    st.subheader("✅ ATS Compatibility")
    st.progress(min(1.0, ats_report.get("ats_score", 0.0)))
    st.write(f"**ATS Score:** {ats_report.get('ats_score', 0.0)*100:.1f}/100")
    if ats_report.get("warnings"):
        st.warning("**Warnings/Suggestions:**\n- " + "\n- ".join(ats_report["warnings"]))
    if ats_report.get("good"):
        st.success("**Good Signs:**\n- " + "\n- ".join(ats_report["good"]))

    # ---------- Download Report ----------
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": model_name,
        "weights": {"semantic": w_sem, "keywords": w_kw},
        "scores": {
            "overall": round(final_score, 2),
            "semantic_similarity": round(sim_score*100, 2),
            "keyword_overlap": round(overlap*100, 2),
            "ats_score": round(ats_report.get("ats_score", 0.0)*100, 2)
        },
        "contact_detected": contact,
        "resume_skills": sorted(list(set(resume_skills))),
        "jd_keywords": sorted(list(set(jd_keywords))),
        "missing_skills": sorted(list(set(missing))),
        "ats_report": ats_report,
        "file_meta": {
            "filetype": parsed.filetype,
            "page_count": parsed.page_count
        }
    }
    buf = io.BytesIO(json.dumps(report, indent=2).encode("utf-8"))
    st.download_button(
        "⬇️ Download JSON Report",
        data=buf,
        file_name="ai_resume_match_report.json",
        mime="application/json",
        use_container_width=True
    )

st.caption("Tip: Keep resumes simple (no text in images), use standard section headers, and tailor keywords to the JD.")
