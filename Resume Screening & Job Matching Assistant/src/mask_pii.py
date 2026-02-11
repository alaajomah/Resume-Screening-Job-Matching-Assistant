"""
- Detects contact info locally (email + phone+ links) using regex to avoid sending PII to the LLM.
- Masks PII in the text by replacing detected emails and valid phone numbers with redacted tokens.
------------------------------------------------------------------------------------------------------ """

import re

EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")

PHONE_RE = re.compile(r"""
(?<!\w)
(?:\+?\d{1,3}[\s\-\.]?)?
(?:\(?\d{2,4}\)?[\s\-\.]?)?
\d{3,4}[\s\-\.]?\d{3,4}
(?!\w)
""", re.VERBOSE)


URL_RE = re.compile(r"""
(?i)\b(
    https?://[^\s<>\]\)"]+
  | www\.[^\s<>\]\)"]+
)\b
""", re.VERBOSE)

# LinkedIn (profile/company + short links)
LINKEDIN_RE = re.compile(r"""
(?i)\b(
    https?://(?:[\w-]+\.)?linkedin\.com/(?:in|pub|company)/[^\s<>\]\)"]+
  | (?:[\w-]+\.)?linkedin\.com/(?:in|pub|company)/[^\s<>\]\)"]+
  | https?://lnkd\.in/[^\s<>\]\)"]+
)\b
""", re.VERBOSE)

# GitHub (profile/repo)
GITHUB_RE = re.compile(r"""
(?i)\b(
    https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?
  | (?:www\.)?github\.com/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?
)\b
""", re.VERBOSE)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")

def _dedupe(items):
    return list(dict.fromkeys([x for x in items if x]))

def _strip_trailing_punct(url: str) -> str:
    return url.rstrip(".,;:!?)\]}>\"'")

def extract_contact_info_local(text: str) -> dict:
    text = text or ""

    emails = _dedupe(EMAIL_RE.findall(text))

    phones = []
    for m in PHONE_RE.finditer(text):
        raw = m.group(0).strip()
        d = _digits(raw)

        if not (8 <= len(d) <= 15):
            continue
        if re.fullmatch(r"(19|20)\d{2}(19|20)\d{2}", d):
            continue

        cleaned = re.sub(r"\s+", " ", raw).strip().replace(".", "-")
        phones.append(cleaned)
    phones = _dedupe(phones)

    linkedin_links = [_strip_trailing_punct(x) for x in LINKEDIN_RE.findall(text)]
    github_links = [_strip_trailing_punct(x) for x in GITHUB_RE.findall(text)]

    all_urls = [_strip_trailing_punct(x) for x in URL_RE.findall(text)]
    all_urls = _dedupe(all_urls)

    specialized = set(map(str.lower, _dedupe(linkedin_links + github_links)))
    other_links = [u for u in all_urls if u.lower() not in specialized]

    return {
        "email": emails[0] if emails else "N/A",
        "phone": phones[0] if phones else "N/A",
        "linkedin": linkedin_links[0] if linkedin_links else "N/A",
        "github": github_links[0] if github_links else "N/A",
        "all_emails": emails,
        "all_phones": phones,
        "all_linkedin": _dedupe(linkedin_links),
        "all_github": _dedupe(github_links),
        "other_links": other_links,
    }


def mask_pii(text: str) -> str:
    if not text:
        return text

    masked = EMAIL_RE.sub("EMAIL_REDACTED", text)

    def _mask_phone(match: re.Match) -> str:
        candidate = match.group(0)
        d = _digits(candidate)

        if not (8 <= len(d) <= 15):
            return candidate
        if re.fullmatch(r"(19|20)\d{2}(19|20)\d{2}", d):
            return candidate
        return "PHONE_REDACTED"

    masked = PHONE_RE.sub(_mask_phone, masked)

    masked = LINKEDIN_RE.sub("LINKEDIN_REDACTED", masked)
    masked = GITHUB_RE.sub("GITHUB_REDACTED", masked)

    def _mask_url(match: re.Match) -> str:
        url = match.group(0)
        url_clean = _strip_trailing_punct(url)
        return "URL_REDACTED" + (url[len(url_clean):] if len(url_clean) < len(url) else "")

    masked = URL_RE.sub(_mask_url, masked)

    return masked
