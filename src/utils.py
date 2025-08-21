
import re
WHITESPACE_RE = re.compile(r"[ \t\u00A0]+")
NEWLINES_RE = re.compile(r"\n{3,}")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    text = NEWLINES_RE.sub("\n\n", text)
    return text.strip()

def guess_section(text: str) -> str:
    headings = [
        "Financial Policy Objectives and Strategies Statement",
        "Maintain a Balanced Budget over the economic cycle",
        "Maintain the Capital Infrastructure of the Territory",
        "Net Interest",
        "Taxation as a proportion of GSP",
        "Net Assets",
        "Superannuation liabilities"
    ]
    for h in headings:
        if h.lower() in text.lower():
            return h
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line[:80] if first_line else "Policy Document"
