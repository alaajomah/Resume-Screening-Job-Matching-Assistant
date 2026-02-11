# Resume-Screening-Job-Matching-Assistant
## Scenario Chosen

This project delivers an **AI Resume Screening & Job Matching Assistant** that supports early-stage hiring decisions by converting unstructured hiring documents into **evidence-backed, structured outputs**. Recruiters upload a candidate CV and a Job Description (JD) in common formats (PDF/DOCX/TXT/PNG/JPG/JPEG). The system then executes three coordinated stages:

1. **JD Understanding**  
   Extracts role responsibilities, required skills/tools, and *must-have vs. nice-to-have* signals directly from the JD text.

2. **CV Understanding**  
   Extracts the candidate’s education, experience, and skills into a standardized structure grounded **only** in what is explicitly stated.

3. **Matching & Recommendations**  
   Performs semantic + keyword alignment between CV and JD to generate a match score, highlight missing requirements, and produce recommended interview questions tailored to gaps and verified strengths.

To ensure responsible use, the system applies **PII masking** before any model call, enforces strict **“no guessing”** rules, and surfaces **verbatim evidence snippets** to make each screening outcome explainable and auditable. Overall, the assistant functions as a **decision-support tool** that improves screening consistency and speed, while keeping final hiring judgment with **human reviewers**.

---

## Target Users

- **Recruiters / HR Coordinators**: accelerate first-pass screening, reduce manual triage time, and ensure consistent shortlisting criteria across candidates.
- **Hiring Managers / Technical Leads**: validate candidate-job alignment using evidence snippets and structured requirement coverage (must-have vs nice-to-have).
- **Talent Operations / HR Analysts (optional)**: export structured JSON for reporting, pipeline analytics, and quality audits of screening decisions.

---

## User Flow

1. **Upload Job Description (PDF/DOCX/TXT/PNG/JPG/JPEG)**  
   The recruiter uploads the JD. The system extracts its text and checks readability (e.g., flags scanned/empty content).

2. **JD Extraction (JD Prompt)**  
   The JD is parsed into structured fields, separating:
   - Must-have requirements  
   - Nice-to-have (preferred/bonus) requirements  
   - Responsibilities and tools/technologies (explicit only)

3. **Upload CV (PDF/DOCX/TXT/PNG/JPG/JPEG)**  
   The recruiter uploads a candidate CV. The system validates the file type and extracts raw text.

4. **PII Masking**  
   The system detects and masks personal identifiers (e.g., name, phone, email, address, links) before sending text to the model, reducing privacy risk and bias signals.

5. **CV Extraction (CV Prompt)**  
   The masked CV text is processed to extract structured information (education, experience, skills when explicitly stated).

6. **Matching & Scoring (Match Prompt)**  
   The system aligns JD extracted requirements with the CV content and produces:
   - An overall match score  
   - Matched requirements (with evidence)  
   - Missing must-have requirements and partial matches  

   **Scoring weights:** 80% must-have requirements + 20% nice-to-have requirements to reflect hiring priority and prevent “bonus skills” from outweighing core requirements.

   If evidence is weak or files are low quality, the system **escalates** by requesting a clearer document rather than guessing.

7. **Suggested Interview Questions**  
   The system generates role-specific interview questions based on:
   - Verifying must-have requirements (highest priority)
   - Confirming nice-to-have advantages
   - Exploring responsibilities through scenarios

---

## System Architecture

1. **Input Layer (UI/API)**  
   Streamlit uploads (JD + CV single/batch), user controls, session state.

2. **Pre-Processing Layer (Text & Parsing)**  
   Text extraction (`extract_text_auto`: PDF/DOCX/TXT/OCR), document classification (`classify_document`), local contact/name heuristics.

3. **Safety Layer (Privacy + Injection Defense)**  
   PII masking (`mask_pii`) + prompt-injection removal/sanitization (`sanitize_prompt_injection_spans`) + UI warnings/escalation.

4. **Prompt Layer (Templates + Rules)**  
   Load prompt blocks (SYSTEM/USER for CV, JD, Match) + schema patching (e.g., add `candidate_name`).

5. **Model Layer (LLM)**  
   `gpt-4o-mini` via `call_llm_json` for: CV extraction, JD extraction, and matching.

6. **Output Layer (Presentation)**  
   Render structured results in the UI (tables, score, matched/missing requirements).

7. **Logging & Evaluation Layer**  
   Current: session + batch summary table; extendable to persistent logs and offline metrics.

![App Screenshot](https://github.com/alaajomah/Resume-Screening-Job-Matching-Assistant/blob/main/System%20Architecture%20Diagram%20.drawio.png)



## OCR Setup (Windows) — Required for Scanned PDFs / Images

If you want OCR (for scanned PDFs or image files), you must install **Tesseract OCR** on Windows.

1) Download and install Tesseract from this release:
https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe

2) After installation, make sure your project knows the Tesseract executable path.
 copy the path and paste it in text_extractor.py file

```env


## How to Run

### 1) Clone the repository
```bash
git clone https://github.com/alaajomah/Resume-Screening-Job-Matching-Assistant.git   
cd Resume-Screening-Job-Matching-Assistant

### 2) Create and activate a virtual environment
python -m venv .venv

### 3) Activate
python -m venv .venv
or from command prompt
.\.venv\Scripts\activate.bat

### 4)Install dependencies
pip install -r requirements.txt

### 5) create .env and put inside it the open ai key
OPENAI_API_KEY=open ai key

### 6) Run the app
streamlit run app.py



