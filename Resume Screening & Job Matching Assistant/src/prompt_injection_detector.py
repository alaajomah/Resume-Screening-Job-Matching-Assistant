import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

# =========================
# 1) Block-level patterns (remove big instruction blocks)
# =========================
BLOCK_PATTERNS = [
    # Prompt blocks
    r"(?is)\bBEGIN\s+PROMPT\b.*?(?:\bEND\s+PROMPT\b|$)",
    r"(?is)\bBEGIN\s+INSTRUCTIONS?\b.*?(?:\bEND\s+INSTRUCTIONS?\b|$)",
    r"(?is)\bSYSTEM\s+PROMPT\b.*?(?:\n{2,}|$)",
    r"(?is)\bDEVELOPER\s+MESSAGE\b.*?(?:\n{2,}|$)",
    # Markdown code fences that often carry prompt/code instructions
    r"(?is)```.*?```",
    # XML-ish / tag blocks sometimes used for jailbreaks
    r"(?is)<\s*(system|developer|instructions?)\s*>.*?<\s*/\s*\1\s*>",
]

# =========================
# 2) Span-level patterns (replace just the malicious spans)
# =========================
SPAN_PATTERNS = [
    # Core instruction attacks
    r"(?i)\bignore\b.{0,120}\b(previous|all)\b.{0,40}\binstructions?\b",
    r"(?i)\b(disregard|bypass|override)\b.{0,80}\b(instructions?|rules|policy|filters?)\b",
    r"(?i)\bdo\s+not\s+follow\b.{0,60}\b(instructions?|rules)\b",
    r"(?i)\bfollow\b.{0,80}\b(these|my)\b.{0,20}\b(instructions?|rules)\b",
    r"(?i)\byou\s+are\s+(chatgpt|an?\s*ai|a\s*large\s*language\s*model)\b",
    r"(?i)\b(system|developer)\s+(prompt|message)\b",
    r"(?i)\b(reveal|show|print|expose)\b.{0,60}\b(prompt|policy|system|developer|rules)\b",
    r"(?i)\boutput\b.{0,30}\b(only|just)\b",
    r"(?i)\breturn\b.{0,30}\b(only|just)\b",
    r"(?i)\bjson\b.{0,30}\b(schema|format|only)\b",
    r"(?i)\bfunction\s*call\b|\btool\s*call\b|\btools?\b.{0,40}\bcall\b",
    r"(?i)\bexecute\b.{0,40}\b(command|code|script)\b",
    r"(?i)\bprompt\s*injection\b|\bjailbreak\b",
    # "Role:" / "Goal:" instruction-like templates embedded
    r"(?i)\brole\s*:\s*.*?(?=$|\n)",
    r"(?i)\bgoals?\s*:\s*.*?(?=$|\n)",
]

# =========================
# 3) Tail-cut triggers (cut remainder of line after strong trigger)
#    This is powerful when injection is appended after valid CV content.
# =========================
TAIL_TRIGGERS = [
    r"(?i)\bignore\b",
    r"(?i)\bdisregard\b",
    r"(?i)\boverride\b",
    r"(?i)\bsystem\s+prompt\b",
    r"(?i)\bdeveloper\s+message\b",
    r"(?i)\byou\s+are\s+chatgpt\b",
    r"(?i)\breturn\s+only\b",
    r"(?i)\boutput\s+only\b",
    r"(?i)\breveal\b",
    r"(?i)\btools?\b",
    r"(?i)\bfunction\s*call\b",
    r"(?i)\bBEGIN\s+PROMPT\b",
]

# Extra: keep CV lines safe if they are just section headings (avoid false positives)
SAFE_SECTION_RE = re.compile(r"(?i)^\s*(education|experience|work experience|skills|projects|summary|objective)\s*[:\-]?\s*$")

# Compile
BLOCK_RE = [re.compile(p) for p in BLOCK_PATTERNS]
SPAN_RE  = [re.compile(p) for p in SPAN_PATTERNS]
TAIL_RE  = [re.compile(p) for p in TAIL_TRIGGERS]

@dataclass
class SanitizeResult:
    sanitized_text: str
    detected: bool
    removed_spans: List[Dict[str, str]]  # {"text":..., "reason":...}
    score: int
    severity: str

def sanitize_prompt_injection_spans(text: str) -> SanitizeResult:
    removed: List[Dict[str, str]] = []
    score = 0
    out = text

    # ---- Step A: Remove big blocks
    for pat in BLOCK_RE:
        def _block_repl(m):
            nonlocal score
            snippet = m.group(0)
            removed.append({"text": snippet, "reason": "instruction_block"})
            score += 5
            return "[REMOVED_INJECTION_BLOCK]"
        out = pat.sub(_block_repl, out)

    # ---- Step B: Span replace across whole text (precise)
    for pat in SPAN_RE:
        def _span_repl(m):
            nonlocal score
            snippet = m.group(0)
            # skip if it's exactly a safe section header
            if SAFE_SECTION_RE.match(snippet.strip()):
                return snippet
            removed.append({"text": snippet, "reason": "injection_span"})
            score += 3
            return "[REMOVED_INJECTION]"
        out = pat.sub(_span_repl, out)

    # ---- Step C: Tail-cut per line (preserve left part)
    lines = out.splitlines()
    new_lines = []
    for line in lines:
        original_line = line
        if SAFE_SECTION_RE.match(line.strip()):
            new_lines.append(line)
            continue

        cut_idx = None
        cut_reason = None
        for trig in TAIL_RE:
            m = trig.search(line)
            if m:
                # Only cut if there's "instruction-like" punctuation/phrasing nearby
                # (reduces false positives on words like "tools" in actual CV)
                window = line[m.start():m.start()+120]
                if re.search(r"(?i)\b(instruction|rules|prompt|return|output|reveal|ignore|follow)\b", window):
                    cut_idx = m.start()
                    cut_reason = "tail_cut_after_trigger"
                    break

        if cut_idx is not None and cut_idx > 0:
            kept = line[:cut_idx].rstrip()
            tail = line[cut_idx:].strip()
            if tail:
                removed.append({"text": tail, "reason": cut_reason})
                score += 2
            new_lines.append((kept + " [REMOVED_INJECTION_TAIL]").rstrip())
        else:
            new_lines.append(original_line)

    out = "\n".join(new_lines)

    detected = score > 0
    severity = "low"
    if score >= 8:
        severity = "high"
    elif score >= 3:
        severity = "medium"

    return SanitizeResult(
        sanitized_text=out,
        detected=detected,
        removed_spans=removed,
        score=score,
        severity=severity
    )
