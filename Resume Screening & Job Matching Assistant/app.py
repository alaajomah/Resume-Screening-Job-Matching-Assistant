# app.py
import os
import json
import re
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import streamlit as st

from src.text_extractor import extract_text_auto
from src.text_classifier import classify_document, DocClassResult
from src.mask_pii import mask_pii, extract_contact_info_local
from src.prompt_injection_detector import sanitize_prompt_injection_spans, SanitizeResult
from src.llm import get_openai_client, call_llm_json


# ----------------------------
# UI config
# ----------------------------
st.set_page_config(
    page_title="Resume Screening & Matching",
    page_icon="📝",
    layout="wide"
)

MODEL_NAME = "gpt-4o-mini"


# ----------------------------
# Prompt loaders
# ----------------------------
def _extract_triple_quoted_var(file_text: str, var_name: str) -> str:
    """Extract content of VAR=\"\"\" ... \"\"\" from a txt file."""
    pattern = rf"{re.escape(var_name)}\s*=\s*\"\"\"(.*?)\"\"\""
    m = re.search(pattern, file_text, re.DOTALL)
    if not m:
        raise ValueError(f"Cannot find {var_name} in prompt file.")
    return m.group(1).strip()


def load_prompts() -> Dict[str, str]:
    base = Path("src/prompts")
    cv_file = (base / "cv_extractor_prompt.txt").read_text(encoding="utf-8", errors="ignore")
    jd_file = (base / "jd_extractor_prompt.txt").read_text(encoding="utf-8", errors="ignore")
    match_file = (base / "match_prompt.txt").read_text(encoding="utf-8", errors="ignore")

    return {
        "CV_SYSTEM": _extract_triple_quoted_var(cv_file, "CV_SYSTEM"),
        "CV_USER": _extract_triple_quoted_var(cv_file, "CV_USER"),
        "JD_SYSTEM": _extract_triple_quoted_var(jd_file, "JD_SYSTEM"),
        "JD_USER": _extract_triple_quoted_var(jd_file, "JD_USER"),
        "MATCH_SYSTEM": _extract_triple_quoted_var(match_file, "match_system"),
        "MATCH_USER": _extract_triple_quoted_var(match_file, "match_user"),
    }


@st.cache_resource
def get_cached_prompts() -> Dict[str, str]:
    return load_prompts()


# ----------------------------
# Patch CV prompt to return candidate_name (LLM)
# NOTE: You still want name in UI. But since you mask text before LLM,
# the LLM may only see [NAME]. So we ALSO extract name locally (below).
# ----------------------------
def patch_cv_user_schema_add_candidate_name(cv_user: str) -> str:
    """
    If schema doesn't include candidate_name, inject it after candidate_summary.
    This is only in the prompt text, NOT the UI.
    """
    if '"candidate_name"' in cv_user:
        return cv_user

    injected = '"candidate_name": "string|null",\n  '
    cv_user = re.sub(
        r'("candidate_summary"\s*:\s*\[.*?\]\s*,\s*)',
        r'\1' + injected,
        cv_user,
        flags=re.DOTALL
    )
    return cv_user


# ----------------------------
# Small helpers
# ----------------------------
def save_uploaded_file_to_tmp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def badge(text: str, kind: str = "info") -> str:
    """kind: info|ok|warn|err"""
    styles = {
        "info": "background:#eef2ff;color:#3730a3;",
        "ok":   "background:#ecfdf5;color:#065f46;",
        "warn": "background:#fffbeb;color:#92400e;",
        "err":  "background:#fef2f2;color:#991b1b;"
    }
    style = styles.get(kind, styles["info"])
    return f'<span style="padding:3px 10px;border-radius:999px;font-size:12px;{style}">{text}</span>'


def method_to_ocr_indicator(extract_res: Dict[str, Any]) -> Tuple[bool, str]:
    """Returns: (is_ocr, label)"""
    method = extract_res.get("method", "")
    is_scanned = bool(extract_res.get("is_scanned_pdf", False))

    if method == "image_ocr":
        return True, "Image OCR"
    if method == "pdf_ocr" and is_scanned:
        return True, "Scanned PDF (OCR)"
    if method == "pdf_text":
        return False, "Digital PDF (Text)"
    if method == "docx":
        return False, "DOCX"
    if method == "txt":
        return False, "TXT"
    return False, method or "Unknown"


def summarize_classifier(cls: DocClassResult) -> str:
    return f"{cls.label.upper()} (cv={cls.cv_score:.1f}, jd={cls.jd_score:.1f})"


# ----------------------------
# Local name extractor (so name doesn't become [NAME])
# ----------------------------
NAME_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z\.\-']+(?:\s+[A-Za-z][A-Za-z\.\-']+){1,4}$")

def extract_candidate_name_local(raw_text: str, contact: Dict[str, Any]) -> Optional[str]:
    """
    Heuristic: choose first 1-3 non-empty lines that looks like a person name.
    Avoid lines containing email/phone/url.
    """
    if not raw_text:
        return None

    emails = set((contact.get("emails") or []) if isinstance(contact.get("emails"), list) else [])
    phone = contact.get("phone")
    # In your contact extractor you may return 'email' and 'phone' (singular). handle both:
    if contact.get("email"):
        emails.add(contact["email"])

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    candidates = lines[:8]  # only early lines

    for ln in candidates:
        low = ln.lower()
        if "@" in ln or "http" in low or "www." in low:
            continue
        if phone and phone in ln:
            continue
        if any(e in ln for e in emails):
            continue

        # Remove common trailing titles
        cleaned = re.sub(r"\b(cv|resume)\b", "", ln, flags=re.I).strip()

        # Must be short-ish
        if len(cleaned) > 50:
            continue

        # Matches "First Last" style
        if NAME_LINE_RE.match(cleaned):
            return cleaned

    return None


def pick_display_name(llm_name: Optional[str], local_name: Optional[str]) -> str:
    """
    Prefer LLM name if it's not masked; otherwise fallback to local.
    """
    llm_name = (llm_name or "").strip()
    local_name = (local_name or "").strip()

    if llm_name and llm_name not in ("[NAME]", "NAME", "null", "None"):
        return llm_name
    if local_name:
        return local_name
    return "Not available"


# ----------------------------
# LLM wrappers
# ----------------------------
def llm_extract_cv(client, prompts: Dict[str, str], sanitized_text: str) -> Dict[str, Any]:
    cv_user = patch_cv_user_schema_add_candidate_name(prompts["CV_USER"])
    user_prompt = cv_user.format(resume_text=sanitized_text)
    return call_llm_json(
        client=client,
        model=MODEL_NAME,
        system_prompt=prompts["CV_SYSTEM"],
        user_prompt=user_prompt,
        temperature=0.1
    )


def llm_extract_jd(client, prompts: Dict[str, str], jd_text: str) -> Dict[str, Any]:
    user_prompt = prompts["JD_USER"].format(job_description=jd_text)
    return call_llm_json(
        client=client,
        model=MODEL_NAME,
        system_prompt=prompts["JD_SYSTEM"],
        user_prompt=user_prompt,
        temperature=0.1
    )


def llm_match(client, prompts: Dict[str, str], cv_masked_text: str, jd_extracted: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = prompts["MATCH_USER"].format(
        cv_text=cv_masked_text,
        jd_extracted_json=json.dumps(jd_extracted, ensure_ascii=False)
    )
    return call_llm_json(
        client=client,
        model=MODEL_NAME,
        system_prompt=prompts["MATCH_SYSTEM"],
        user_prompt=user_prompt,
        temperature=0.1
    )


# ----------------------------
# UI rendering (NO JSON)
# ----------------------------
def render_cv_analysis(
    cv_out: Dict[str, Any],
    contact_local: Dict[str, Any],
    local_candidate_name: Optional[str] = None,
    use_work_expanders: bool = True
):
    # Personal info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("👤 Personal Info")
        real_name = pick_display_name(cv_out.get("candidate_name"), local_candidate_name)
        st.write(f"**Name:** {real_name}")

    with col2:
        st.subheader("📧 Email")
        st.write(f"**Email:** {contact_local.get('email', 'N/A')}")

    with col3:
        st.subheader("📱 Phone")
        st.write(f"**Phone:** {contact_local.get('phone', 'N/A')}")

    st.divider()

    # Candidate summary
    st.subheader("🧾 Professional Summary")
    summary = cv_out.get("candidate_summary") or []
    if summary:
        for s in summary:
            st.write(f"• {s}")
    else:
        st.write("No summary found.")

    # Education
    st.subheader("🎓 Education")
    edu = cv_out.get("education_history") or []
    if edu:
        df = pd.DataFrame([{
            "Degree": e.get("degree", ""),
            "Institution": e.get("institution", ""),
            "Year": e.get("year", "")
        } for e in edu])
        st.dataframe(df, use_container_width=True, hide_index=True, height=280)
    else:
        st.info("No education extracted.")

    # Work Experience
    st.subheader("🧰 Work Experience")
    work = cv_out.get("work_experience") or []
    if work:
        for i, w in enumerate(work, start=1):
            title = f"{i}) {w.get('role','')} — {w.get('company','')}".strip()

            if use_work_expanders:
                with st.expander(title, expanded=False):
                    st.write(f"**Duration:** {w.get('duration','')}")
                    ach = w.get("key_achievements") or []
                    if ach:
                        st.write("**Achievements:**")
                        for a in ach:
                            st.write(f"• {a}")
                    else:
                        st.write("No achievements found.")
            else:
                st.markdown(f"**{title}**")
                st.write(f"**Duration:** {w.get('duration','')}")
                ach = w.get("key_achievements") or []
                if ach:
                    st.write("**Achievements:**")
                    for a in ach:
                        st.write(f"• {a}")
                else:
                    st.write("No achievements found.")
                st.divider()
    else:
        st.info("No work experience extracted.")

    # Skills
    st.subheader("🧩 Skills")
    skills = cv_out.get("skills") or []
    if skills:
        skill_list = [s.get("skill", "") for s in skills if s.get("skill")]
        st.write(", ".join(skill_list))
    else:
        st.info("No skills extracted.")

    # Issues
    issues = cv_out.get("issues") or []
    if issues:
        st.subheader("⚠️ Notes / Issues")
        for it in issues:
            st.write(f"- **{it.get('type','other')}**: {it.get('detail','')}")


def render_match_results(match_out: Dict[str, Any]):
    st.subheader("🎯 Match Result")

    score = match_out.get("overall_fit_score", 0)
    fit = match_out.get("fit_level", "N/A")
    role = match_out.get("role_title")
    seniority = match_out.get("seniority_range")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fit Score", int(score))
    c2.metric("Fit Level", fit)
    c3.metric("Role Title", role or "—")
    c4.metric("Seniority", seniority or "—")

    st.progress(min(max(int(score), 0), 100))
    st.divider()

    # Breakdown
    bd = match_out.get("requirements_breakdown", {}) or {}
    mh = bd.get("must_have_verifiable", {}) or {}
    nh = bd.get("nice_to_have_verifiable", {}) or {}
    bs = bd.get("behavioral_skills", {}) or {}

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("**Must-have (Verifiable)**")
        st.write(f"Total: {mh.get('total',0)}")
        st.write(f"Matched: {mh.get('matched',0)}")
        st.write(f"Missing: {mh.get('missing',0)}")

    with colB:
        st.markdown("**Nice-to-have (Verifiable)**")
        st.write(f"Total: {nh.get('total',0)}")
        st.write(f"Matched: {nh.get('matched',0)}")
        st.write(f"Missing: {nh.get('missing',0)}")

    with colC:
        st.markdown("**Behavioral Skills** (not scored)")
        st.write(f"Total: {bs.get('total',0)}")
        st.write(f"Evidence found: {bs.get('found_contextual_evidence',0)}")
        st.write(f"Interview needed: {bs.get('needs_interview_assessment',0)}")

    st.divider()

    matched = match_out.get("matched_requirements") or []
    if matched:
        st.subheader("✅ Matched Requirements")
        df = pd.DataFrame([{
            "ID": r.get("req_id"),
            "Requirement": r.get("requirement"),
            "Priority": r.get("priority"),
            "Type": r.get("requirement_type"),
            "Matched Component": r.get("matched_component"),
            "Match Type": r.get("match_type"),
            "Evidence": r.get("evidence_from_cv"),
            "Notes": r.get("notes"),
        } for r in matched])
        st.dataframe(df, use_container_width=True, hide_index=True, height=420)

    missing = match_out.get("missing_requirements") or []
    if missing:
        st.subheader("❌ Missing Requirements")
        df2 = pd.DataFrame([{
            "ID": r.get("req_id"),
            "Requirement": r.get("requirement"),
            "Priority": r.get("priority"),
            "Type": r.get("requirement_type"),
            "Reason": r.get("reason"),
        } for r in missing])
        st.dataframe(df2, use_container_width=True, hide_index=True, height=300)

    strengths = match_out.get("contextual_strengths") or []
    if strengths:
        st.subheader("🌟 Contextual Strengths (Bonus)")
        for s in strengths:
            with st.expander(s.get("skill", "Strength"), expanded=False):
                st.write(f"**Evidence:** {s.get('evidence','')}")
                st.write(f"**Impact:** {s.get('impact','')}")
                st.write(f"**Note:** {s.get('note','')}")

    interview = match_out.get("interview_assessment_needed") or []
    if interview:
        st.subheader("🧪 Suggested Interview Questions")
        for q in interview:
            with st.expander(q.get("skill", "Interview"), expanded=False):
                st.write(f"**Priority:** {q.get('priority','')}")
                st.write(f"**Question:** {q.get('suggested_question','')}")
                st.write(f"**Focus:** {q.get('assessment_focus','')}")

    resp_cov = match_out.get("responsibilities_coverage") or []
    if resp_cov:
        st.subheader("📌 Responsibilities Coverage")
        df3 = pd.DataFrame([{
            "Responsibility": r.get("item"),
            "Covered": "Yes" if r.get("covered") else "No",
            "Evidence": r.get("evidence_from_cv"),
        } for r in resp_cov])
        st.dataframe(df3, use_container_width=True, hide_index=True, height=320)

    issues = match_out.get("issues") or []
    if issues:
        st.subheader("⚠️ Notes")
        for it in issues:
            st.write(f"- **{it.get('type','other')}**: {it.get('detail','')}")


# ----------------------------
# Processing pipeline
# ----------------------------
def process_jd_upload(client, prompts: Dict[str, str], uploaded_jd) -> Tuple[bool, str]:
    tmp_path = save_uploaded_file_to_tmp(uploaded_jd)
    try:
        extract_res = extract_text_auto(
            tmp_path,
            ocr_lang=st.session_state.get("ocr_lang", "eng"),
            ocr_dpi=300
        )
        text = extract_res.get("text", "") or ""
        cls = classify_document(text)

        if cls.label != "jd":
            return False, f"This file is not a JD. Classification: {summarize_classifier(cls)}"

        jd_out = llm_extract_jd(client, prompts, text)

        st.session_state["jd_ready"] = True
        st.session_state["jd_text"] = text
        st.session_state["jd_extract"] = jd_out
        st.session_state["jd_extract_method"] = extract_res
        st.session_state["jd_class"] = cls
        return True, "JD saved in session ✅"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def process_single_cv(client, prompts: Dict[str, str], uploaded_cv) -> Dict[str, Any]:
    tmp_path = save_uploaded_file_to_tmp(uploaded_cv)
    try:
        extract_res = extract_text_auto(
            tmp_path,
            ocr_lang=st.session_state.get("ocr_lang", "eng"),
            ocr_dpi=300
        )
        raw_text = extract_res.get("text", "") or ""
        cls = classify_document(raw_text)

        out: Dict[str, Any] = {
            "file_name": uploaded_cv.name,
            "extract_res": extract_res,
            "class_res": cls,
            "ok": False,
            "error": None,
        }

        if cls.label != "cv":
            out["error"] = f"This file is not a CV. Classification: {summarize_classifier(cls)}"
            return out

        # local contact + local candidate name (BEFORE masking)
        contact = extract_contact_info_local(raw_text)
        local_name = extract_candidate_name_local(raw_text, contact)

        # mask PII then prompt injection sanitize (LLM receives masked text)
        masked_text = mask_pii(raw_text)
        inj: SanitizeResult = sanitize_prompt_injection_spans(masked_text)

        cv_out = llm_extract_cv(client, prompts, inj.sanitized_text)

        out.update({
            "ok": True,
            "contact_local": contact,
            "local_candidate_name": local_name,
            "injection": inj,
            "cv_out": cv_out,
        })
        return out

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ----------------------------
# Main
# ----------------------------
prompts = get_cached_prompts()
client = get_openai_client()

st.title("📝 Resume Screening & Job Matching Assistant")
st.caption("Upload a JD once, then upload a single CV or a batch")

with st.sidebar:
    if not client:
        st.error("OPENAI_API_KEY was not found. Set it as an environment variable or in Streamlit secrets.")
        st.stop()



    st.divider()
    st.subheader("📌 Upload JD (once)")
    jd_file = st.file_uploader(
        "Upload Job Description",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
        key="jd_uploader"
    )

    if st.button("✅ Save JD to Session", use_container_width=True, disabled=(jd_file is None)):
        ok, msg = process_jd_upload(client, prompts, jd_file)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    if st.session_state.get("jd_ready"):
        st.markdown(badge("JD READY", "ok"), unsafe_allow_html=True)
        ex = st.session_state.get("jd_extract_method", {})
        is_ocr, label = method_to_ocr_indicator(ex)
        st.markdown(badge(label, "warn" if is_ocr else "info"), unsafe_allow_html=True)

        jd_extract = st.session_state.get("jd_extract", {})
        st.write("**Role Title:**", jd_extract.get("role_title") or "—")
        st.write("**Seniority:**", jd_extract.get("seniority_range") or "—")
    else:
        st.markdown(badge("JD NOT READY", "warn"), unsafe_allow_html=True)
        st.caption("Matching will be enabled only after saving the JD.")


# ✅ Session state init (OUTSIDE sidebar, before tabs)
if "last_cv_result" not in st.session_state:
    st.session_state["last_cv_result"] = None
if "last_match_result" not in st.session_state:
    st.session_state["last_match_result"] = None
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None


# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["📄 Single CV", "📚 Batch CVs"])

with tab1:
    st.subheader("📄 Upload a Single CV")

    cv_file = st.file_uploader(
        "Upload CV",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
        key="single_cv"
    )

    if cv_file and st.button("🚀 Analyze CV", type="primary", key="btn_analyze_single"):
        with st.spinner("Processing..."):
            st.session_state["last_cv_result"] = process_single_cv(client, prompts, cv_file)
            st.session_state["last_match_result"] = None

    result = st.session_state.get("last_cv_result")
    if result:
        ex = result["extract_res"]
        cls = result["class_res"]
        is_ocr, label = method_to_ocr_indicator(ex)

        top_cols = st.columns(4)
        top_cols[0].markdown(badge(f"File: {result['file_name']}", "info"), unsafe_allow_html=True)
        top_cols[1].markdown(badge(label, "warn" if is_ocr else "info"), unsafe_allow_html=True)

        kind = "ok" if cls.label == "cv" else ("err" if cls.label == "jd" else "warn")
        top_cols[2].markdown(badge(f"Class: {cls.label.upper()}", kind), unsafe_allow_html=True)

        inj: Optional[SanitizeResult] = result.get("injection")
        if inj and inj.detected:
            sev_kind = "warn" if inj.severity in ("low", "medium") else "err"
            top_cols[3].markdown(badge(f"Prompt Injection: {inj.severity.upper()}", sev_kind), unsafe_allow_html=True)
        else:
            top_cols[3].markdown(badge("Prompt Injection: NONE", "ok"), unsafe_allow_html=True)

        st.divider()

        if not result["ok"]:
            st.error("❌ Wrong file type uploaded in the CV section.")
            st.write(result["error"])
        else:
            render_cv_analysis(
                result["cv_out"],
                result["contact_local"],
                local_candidate_name=result.get("local_candidate_name"),
                use_work_expanders=True
            )

            st.divider()

            if not st.session_state.get("jd_ready"):
                st.warning("Please upload and save a JD from the sidebar first to run matching.")
            else:
                st.subheader("🔁 Match Against JD")

                if st.button("✅ Run Matching", use_container_width=True, key="btn_match_single"):
                    with st.spinner("Matching..."):
                        cv_masked = result["cv_out"].get("masked_resume_text") or ""
                        st.session_state["last_match_result"] = llm_match(
                            client, prompts, cv_masked, st.session_state["jd_extract"]
                        )

                match_out = st.session_state.get("last_match_result")
                if match_out:
                    render_match_results(match_out)


with tab2:
    st.subheader("📚 Upload Batch CVs")

    batch_files = st.file_uploader(
        "Upload multiple CVs",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="batch_cvs"
    )

    run_match = st.checkbox("Run matching after analysis (requires JD saved)", value=True)

    batch_progress = st.progress(0)
    batch_status = st.empty()

    if batch_files and st.button("🚀 Start Batch Processing", type="primary", key="btn_batch"):
        results = []
        rows = []

        total = len(batch_files)
        batch_status.write(f"Starting batch: {total} file(s)")
        batch_progress.progress(0)

        for idx, f in enumerate(batch_files, start=1):
            batch_status.write(f"Processing {idx}/{total}: {f.name}")

            try:
                res = process_single_cv(client, prompts, f)
                ex = res["extract_res"]
                is_ocr, label = method_to_ocr_indicator(ex)
                cls: DocClassResult = res["class_res"]

                inj = res.get("injection")
                inj_text = "none"
                if inj and inj.detected:
                    inj_text = f"{inj.severity} (score={inj.score})"

                match_out = None
                match_done = "—"
                fit_score = "—"
                fit_level = "—"

                if res["ok"] and run_match and st.session_state.get("jd_ready"):
                    cv_masked = res["cv_out"].get("masked_resume_text") or ""
                    match_out = llm_match(client, prompts, cv_masked, st.session_state["jd_extract"])
                    match_done = "✅"
                    fit_score = str(match_out.get("overall_fit_score", "—"))
                    fit_level = str(match_out.get("fit_level", "—"))
                elif res["ok"] and run_match and not st.session_state.get("jd_ready"):
                    match_done = "⚠️ JD not ready"

                results.append({
                    "res": res,
                    "ocr_label": label,
                    "is_ocr": is_ocr,
                    "inj_text": inj_text,
                    "match_out": match_out,
                })

                rows.append({
                    "File": res["file_name"],
                    "Extract": label,
                    "OCR?": "Yes" if is_ocr else "No",
                    "Class": cls.label.upper(),
                    "Injection": inj_text,
                    "CV Extracted": "✅" if res["ok"] else "❌",
                    "Match": match_done,
                    "Fit Score": fit_score,
                    "Fit Level": fit_level,
                    "Error": res.get("error") or ""
                })

            except Exception as e:
                rows.append({
                    "File": f.name,
                    "Extract": "—",
                    "OCR?": "—",
                    "Class": "—",
                    "Injection": "—",
                    "CV Extracted": "❌",
                    "Match": "—",
                    "Fit Score": "—",
                    "Fit Level": "—",
                    "Error": str(e)
                })

            batch_progress.progress(int(idx / total * 100))

        batch_status.success("✅ Batch done")

        st.session_state["batch_results"] = {
            "rows": rows,
            "results": results,
        }

    batch_data = st.session_state.get("batch_results")

    if batch_data:
        st.subheader("📊 File Status Table")
        df = pd.DataFrame(batch_data["rows"])
        st.dataframe(df, use_container_width=True, hide_index=True, height=420)

        st.divider()
        st.subheader("🧾 Extracted CV Analysis (Batch)")

        # ✅ IMPORTANT: No outer expander here (prevents nested expander crash)
        for i, item in enumerate(batch_data["results"], start=1):
            res = item["res"]
            cls = res["class_res"]
            label = item["ocr_label"]
            is_ocr = item["is_ocr"]
            match_out = item["match_out"]

            st.markdown(f"### {i}) {res['file_name']} | {cls.label.upper()} | {label}")

            # indicators row
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(badge(f"File: {res['file_name']}", "info"), unsafe_allow_html=True)
            c2.markdown(badge(label, "warn" if is_ocr else "info"), unsafe_allow_html=True)

            kind = "ok" if cls.label == "cv" else ("err" if cls.label == "jd" else "warn")
            c3.markdown(badge(f"Class: {cls.label.upper()}", kind), unsafe_allow_html=True)

            inj: Optional[SanitizeResult] = res.get("injection")
            if inj and inj.detected:
                sev_kind = "warn" if inj.severity in ("low", "medium") else "err"
                c4.markdown(badge(f"Prompt Injection: {inj.severity.upper()}", sev_kind), unsafe_allow_html=True)
            else:
                c4.markdown(badge("Prompt Injection: NONE", "ok"), unsafe_allow_html=True)

            st.divider()

            if not res["ok"]:
                st.error("Wrong file type uploaded in the CV batch.")
                st.write(res.get("error", ""))
                st.divider()
                continue

            # ✅ CV analysis UI (NO JSON)
            render_cv_analysis(
                res["cv_out"],
                res["contact_local"],
                local_candidate_name=res.get("local_candidate_name"),
                use_work_expanders=False  # avoids nested expanders patterns in batch layout
            )

            # ✅ Match UI (NO JSON)
            if run_match:
                st.divider()
                st.subheader("Matching")
                if not st.session_state.get("jd_ready"):
                    st.warning("JD not saved. Matching skipped.")
                elif match_out:
                    render_match_results(match_out)
                else:
                    st.info("Matching was not generated for this file.")

            st.divider()
