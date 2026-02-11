import re
from dataclasses import dataclass
from typing import List

# =========================================================
# Regex (compiled)
# =========================================================
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.I)
URL_RE   = re.compile(r"\b(?:https?://|www\.)\S+\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?)?\d{3,4}[\s\-\.]?\d{3,4}(?!\w)")
DATE_RE  = re.compile(
    r"\b(?:19|20)\d{2}\b|"                                     # 2019
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b|"  # Jan
    r"\b\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})\b",              # 12/01/2023
    re.I
)

# =========================================================
# Helpers
# =========================================================
def normalize_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def count_matches(patterns: List[str], text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text, re.I))

def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")

def _first_nonempty_lines(text: str, n: int = 8) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[:n]

def looks_like_role_title_top(text: str) -> bool:
    """
    Detects if the document starts with a job title line (common in JDs).
    """
    lines = _first_nonempty_lines(text, n=6)
    if not lines:
        return False

    top = lines[0]
    # too long to be a title
    if len(top) > 80:
        return False

    # contains a known role word
    role_word = any(re.search(rf"\b{rw}\b", top, re.I) for rw in JD_ROLE_TITLES)
    if not role_word:
        return False

    # avoid CV name line: if email/phone in top line, probably CV header
    if EMAIL_RE.search(top) or PHONE_RE.search(top):
        return False

    return True

def looks_like_list_block_after_heading(raw_text: str, heading_regex: str, min_lines: int = 4) -> bool:
    """
    Finds heading then checks next ~800 chars for multiple short lines (even without bullets).
    """
    if not raw_text:
        return False

    m = re.search(heading_regex, raw_text, re.I)
    if not m:
        return False

    window = raw_text[m.end(): m.end() + 900]
    lines = [ln.strip() for ln in window.splitlines() if ln.strip()]

    # stop on another heading
    stop_re = re.compile(r"\b(requirements|qualifications|benefits|we offer|about us|job summary|responsibilities)\b", re.I)
    cleaned = []
    for ln in lines:
        if stop_re.search(ln) and len(cleaned) >= 1:
            break
        cleaned.append(ln)

    item_like = sum(1 for ln in cleaned if 10 <= len(ln) <= 160)
    return item_like >= min_lines

# =========================================================
# CV signals
# =========================================================

CV_SECTION_HEADINGS_STRONG = [
    r"\bprofessional summary\b", r"\bsummary\b", r"\bcareer objective\b", r"\bobjective\b",
    r"\beducation\b", r"\bgraduation date\b", r"\bacademic\b",
    r"\bexperience\b", r"\bwork experience\b", r"\bemployment\b",
    r"\binternship\b", r"\btraining\b", r"\bworkshops?\b",
    r"\bprojects?\b", r"\bgraduation project\b",
    r"\bskills?\b", r"\btechnical skills?\b", r"\bsoft skills?\b",
    r"\bcourses?\b", r"\bcertifications?\b",
    r"\blanguages?\b", r"\bmother tongue\b",
    r"\breferences?\b", r"\bavailable when needed\b"
]

CV_CONTACT_LABELS = [
    r"\bphone\b\s*[:\-]", r"\bphone number\b\s*[:\-]",
    r"\be-?mail\b\s*[:\-]", r"\bemail\b\s*[:\-]",
    r"\blocation\b\s*[:\-]", r"\baddress\b\s*[:\-]"
]

CV_DATE_RANGES = [
    r"\b(19|20)\d{2}\s*[-–]\s*(19|20)\d{2}\b",              # 2019–2023
    r"\b(0?\d)/(19|20)\d{2}\s*[-–]\s*(0?\d)/(19|20)\d{2}\b",# 06/2023–11/2023
    r"\b(19|20)\d{2}\s*[-–]\s*(present|current)\b"          # 2023–Present
]

CV_COMPANY_LOCATION_LINE = [
    r"\b[A-Z][A-Za-z0-9&\.\- ]{2,}\b\s+[A-Z][A-Za-z\- ]+\s*[–-]\s*[A-Z][A-Za-z\- ]+\s+\b(19|20)\d{2}\s*[–-]\s*(19|20)\d{2}\b",
    # مثال: Sohool Development Company Riyadh – Saudi Arabia 2017–2021
]

CV_ROLE_TITLE_TOP = [
    r"(?m)^[A-Z][A-Za-z\.\- ]{2,40}$\n^[A-Za-z][A-Za-z0-9/ \-]{2,40}\b(developer|engineer|manager|analyst|designer|intern|consultant)\b"
]

CV_MORE_HEADINGS = [
    r"\babout me\b", r"\bprofile\b", r"\bpersonal profile\b",
    r"\bpersonal information\b", r"\bpersonal details\b",
    r"\bcontact\b", r"\bcontacts\b", r"\baddress\b",
    r"\bdate of birth\b", r"\bdob\b", r"\bage\b",
    r"\bnationality\b", r"\bcitizenship\b",
    r"\bmarital status\b", r"\bsingle\b", r"\bmarried\b", r"\bdivorced\b",
    r"\bplace of birth\b", r"\bborn in\b",
    r"\breligion\b",
    r"\bgender\b", r"\bmale\b", r"\bfemale\b",
    r"\bmilitary service\b",
    r"\bdriving license\b", r"\bdriver'?s license\b",
    r"\breferences\b", r"\breferees\b",
    r"\bhobbies\b", r"\binterests\b", r"\bvolunteer(ing)?\b",
    r"\bobjective\b", r"\bcareer objective\b",
]

CV_PERSONAL_INFO_PATTERNS = [
    r"\bdate of birth\b\s*[:\-]",
    r"\bplace of birth\b\s*[:\-]",
    r"\bnationality\b\s*[:\-]",
    r"\bmarital status\b\s*[:\-]",
    r"\breligion\b\s*[:\-]",
    r"\bgender\b\s*[:\-]",
    r"\baddress\b\s*[:\-]",
]

CV_ACHIEVEMENT_PATTERNS = [
    r"\bachieved\b", r"\bimproved\b", r"\bincreased\b", r"\breduced\b",
    r"\bled\b", r"\bmanaged\b", r"\bdeveloped\b", r"\bdesigned\b",
    r"\bimplemented\b", r"\bdelivered\b",
    r"\b\d+%|\b\d+\s*(users|customers|projects|apps|models)\b"
]

FIRST_PERSON = [
    r"\bi am\b", r"\bmy\b", r"\bi'm\b", r"\bworked on\b", r"\bled\b", r"\bdeveloped\b"
]

EMPLOYMENT_WORDS = [
    r"\bcompany\b", r"\bintern\b", r"\binternship\b", r"\bengineer\b", r"\bdeveloper\b",
    r"\bmanager\b", r"\bassistant\b", r"\bconsultant\b", r"\bteam\b"
]

# =========================================================
# JD signals (STRONGER)
# =========================================================
JD_HEADINGS = [
    r"\bjob description\b", r"\bjob summary\b", r"\brole summary\b",
    r"\babout (the )?role\b", r"\brole overview\b", r"\boverview\b",
    r"\babout us\b", r"\bcompany\b", r"\bwho we are\b",
    r"\bresponsibilities\b", r"\bkey responsibilities\b", r"\bkey duties\b",
    r"\bduties\b", r"\baccountabilities\b", r"\bkey accountabilities\b",
    r"\bwhat you'll do\b", r"\bwhat you will do\b",
    r"\bwhat we're looking for\b", r"\bwho you are\b",

    r"\brequirements\b", r"\bjob requirements\b", r"\brequired skills\b",
    r"\bqualifications\b", r"\bminimum qualifications\b",
    r"\bpreferred qualifications\b", r"\bpreferred skills\b",
    r"\bmust[- ]have\b", r"\bnice[- ]to[- ]have\b",

    r"\bwe offer\b", r"\bbenefits\b", r"\bperks\b",
    r"\bcompensation\b", r"\bsalary\b", r"\bpackage\b",

    r"\bwork location\b", r"\blocation\b",
    r"\bemployment type\b", r"\bcontract\b",
    r"\bfull[- ]time\b", r"\bpart[- ]time\b",
    r"\bhybrid\b", r"\bremote\b", r"\bon[- ]site\b",

    r"\bequal opportunity\b", r"\beeoc\b", r"\bdiversity\b", r"\binclusion\b"
]

JD_INTENT_PHRASES = [
    r"\bwe are looking for\b", r"\bwe are seeking\b", r"\bwe are hiring\b",
    r"\bjoin our team\b", r"\bthe ideal candidate\b", r"\bsuccessful candidate\b",
    r"\bas a\b.+\byou will\b", r"\byou will\b", r"\byou'll\b",
    r"\byou are responsible for\b", r"\byour responsibilities\b",
    r"\breporting to\b", r"\breports to\b",

    r"\bmust have\b", r"\brequired\b", r"\bpreferred\b",
    r"\bnice to have\b", r"\bstrong (advantage|plus)\b",
    r"\bapply\b", r"\bapply now\b", r"\bsubmit\b", r"\bapplication\b",
    r"\bsend your (cv|resume)\b", r"\bhow to apply\b"
]

JD_HR_WORDS = [
    r"\bcandidate\b", r"\bapplicant\b", r"\bposition\b", r"\bvacancy\b",
    r"\brole\b", r"\bjob\b", r"\bopening\b",
    r"\brecruit(ment|er|ing)\b", r"\bhr\b", r"\bhuman resources\b",
    r"\bwork authorization\b", r"\bvisa\b", r"\brelocation\b",
    r"\bnotice period\b", r"\bstart date\b"
]

JD_REQUIREMENT_PATTERNS = [
    # years of experience variants
    r"\b\d+\s*(?:to|-|–)\s*\d+\s*years?\s+of\s+experience\b",   # 0 to 3 years
    r"\b\d+\+?\s*(years?|yrs?)\s+of\s+experience\b",
    r"\bminimum\s+\d+\s*(years?|yrs?)\b",
    r"\bexperience\s+in\b",
    r"\bproficien(t|cy)\s+in\b",
    r"\bability\s+to\b",
    r"\bstrong\s+(knowledge|understanding)\s+of\b",
    r"\bfamiliar(ity)?\s+with\b",

    # education requirement patterns
    r"\bdegree\b.*\b(preferred|required)\b",
    r"\b(bachelor|master|b\.sc|bs|bsc|m\.sc|msc)\b.*\b(preferred|required)\b",

    r"\brequired\b",
    r"\bmust\b"
]

# Common role title words (helps when JD is just a title + short text)
JD_ROLE_TITLES = [
    r"developer", r"engineer", r"designer", r"analyst", r"specialist",
    r"manager", r"consultant", r"intern", r"lead", r"architect"
]


# =========================================================
# Result type
# =========================================================
@dataclass
class DocClassResult:
    label: str          # "cv" | "jd" | "unknown"
    cv_score: float
    jd_score: float
    reasons: List[str]

# =========================================================
# Classifier
# =========================================================
def classify_document(text: str, min_chars: int = 200) -> DocClassResult:
    t = normalize_text(text)
    reasons: List[str] = []

    if len(t) < min_chars:
        return DocClassResult("unknown", 0.0, 0.0, [f"Too short ({len(t)} chars)"])

    cv_score = 0.0
    jd_score = 0.0

    # ---------------- CV scoring ----------------
    email_found = bool(EMAIL_RE.search(t))
    url_found   = bool(URL_RE.search(t))

    phone_found = False
    for m in PHONE_RE.finditer(t):
        if len(_digits(m.group(0))) >= 8:
            phone_found = True
            break

        # --- Strong CV: section headings density
    sec_hits = count_matches(CV_SECTION_HEADINGS_STRONG, t)
    if sec_hits:
        cv_score += 1.6 * sec_hits
        reasons.append(f"CV: strong section headings x{sec_hits}")

    # --- Strong CV: contact labels (Location/Phone/E-mail) حتى لو ما انمسكت بالـregex
    label_hits = count_matches(CV_CONTACT_LABELS, t)
    if label_hits:
        cv_score += 2.0
        reasons.append(f"CV: contact labels present x{label_hits}")

    # --- Strong CV: date ranges (2019–2023 / 06/2023–11/2023)
    range_hits = count_matches(CV_DATE_RANGES, t)
    if range_hits:
        cv_score += 2.0
        reasons.append(f"CV: date ranges x{range_hits}")

    # --- Strong CV: "Company City – Country Year-Year" line (CV4 style)
    locline_hits = count_matches(CV_COMPANY_LOCATION_LINE, text or "")
    if locline_hits:
        cv_score += 2.5
        reasons.append("CV: company-location-year line detected")

    # --- Strong CV: name + role title in first lines (CV header style)
    if re.search(CV_ROLE_TITLE_TOP[0], (text or "").strip(), re.I):
        cv_score += 2.0
        reasons.append("CV: header name+title pattern detected")


    identity_hits = sum([email_found, phone_found, url_found])
    if identity_hits:
        cv_score += 3.0 * identity_hits
        reasons.append(f"CV: identity signals x{identity_hits}")

    more_cv_headings = count_matches(CV_MORE_HEADINGS, t)
    if more_cv_headings:
        cv_score += 1.0 * more_cv_headings
        reasons.append(f"CV: personal headings x{more_cv_headings}")

    personal_info_hits = count_matches(CV_PERSONAL_INFO_PATTERNS, t)
    if personal_info_hits:
        cv_score += 2.5
        reasons.append(f"CV: personal-info patterns x{personal_info_hits}")

    ach_hits = count_matches(CV_ACHIEVEMENT_PATTERNS, t)
    if ach_hits >= 2:
        cv_score += 1.5
        reasons.append("CV: achievement-style statements")

    date_hits = len(DATE_RE.findall(t))
    employment_hits = count_matches(EMPLOYMENT_WORDS, t)
    if date_hits >= 3 and employment_hits >= 1:
        cv_score += 3.0
        reasons.append("CV: chronological signals (dates + employment words)")

    first_person_hits = count_matches(FIRST_PERSON, t)
    if first_person_hits >= 2:
        cv_score += 1.5
        reasons.append("CV: first-person narrative")

    # ---------------- JD scoring ----------------
    jd_heading_hits = count_matches(JD_HEADINGS, t)
    if jd_heading_hits:
        jd_score += 2.0 * jd_heading_hits
        reasons.append(f"JD: headings x{jd_heading_hits}")

    # Job title / vacancy line
    if re.search(r"(?m)^(hiring|vacancy|position|job)\s*[:\-]", text or "", re.I):
        jd_score += 2.5
        reasons.append("JD: hiring/vacancy title line")

    jd_intent_hits = count_matches(JD_INTENT_PHRASES, t)
    if jd_intent_hits:
        jd_score += 2.0 * jd_intent_hits
        reasons.append(f"JD: intent phrases x{jd_intent_hits}")
        # ---- Strong JD: role title on top
    if looks_like_role_title_top(text):
        jd_score += 2.5
        reasons.append("JD: role title at top")

    # ---- Strong JD: responsibilities/requirements list blocks (even without bullets)
    if looks_like_list_block_after_heading(text, r"\b(key\s+)?responsibilities\b|\bduties\b|\baccountabilities\b", min_lines=4):
        jd_score += 3.0
        reasons.append("JD: responsibilities list-block detected")

    if looks_like_list_block_after_heading(text, r"\brequirements\b|\bqualifications\b|\bminimum qualifications\b|\bpreferred qualifications\b", min_lines=4):
        jd_score += 3.0
        reasons.append("JD: requirements list-block detected")

    # ---- Strong JD: requirement patterns
    req_hits = count_matches(JD_REQUIREMENT_PATTERNS, t)
    if req_hits:
        jd_score += 2.5
        reasons.append(f"JD: requirement patterns x{req_hits}")

    # ---- Better list detection (bullets OR numbering OR colon-lines)
    bullet_hits = len(re.findall(r"(?:\n|^)\s*(?:[-•*]|\d+\.)\s+\S", text or ""))
    colon_list_hits = len(re.findall(r"(?m)^[A-Z][A-Za-z /&]{2,25}:\s+\S", text or ""))  # e.g. "Requirements: ..."
    if bullet_hits >= 5 or colon_list_hits >= 2:
        jd_score += 2.0
        reasons.append(f"JD: list structure (bullets={bullet_hits}, colon_lines={colon_list_hits})")


    hr_hits = count_matches(JD_HR_WORDS, t)
    if hr_hits:
        jd_score += 1.0 * hr_hits
        reasons.append(f"JD: HR vocab x{hr_hits}")



    # Application CTA
    if re.search(r"\b(apply|application|submit|send your (cv|resume))\b", t, re.I):
        jd_score += 2.0
        reasons.append("JD: application call-to-action")

    # ---------------- Decision ----------------
    margin = 2.0
    min_win = 3.0

    if cv_score >= min_win and (cv_score - jd_score) >= margin:
        label = "cv"
    elif jd_score >= min_win and (jd_score - cv_score) >= margin:
        label = "jd"
    else:
        label = "unknown"

    return DocClassResult(label, cv_score, jd_score, reasons)

# =========================================================
# Quick test
# =========================================================
if __name__ == "__main__":
    sample = """
    We are seeking a Python Developer.
    Responsibilities:
    - Build APIs
    - Work with cross-functional teams
    Requirements:
    - 2+ years of Python
    - Nice to have: Docker
    Apply now and send your CV.
    """
    print(classify_document(sample))
