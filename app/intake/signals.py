"""Deterministic hard-escalation signals extracted at intake time (PDF 3.2 / 3.4).

The model's classification may *add* a hard category but can never remove one found here, so a
request cannot escape escalation by being rephrased to sound routine ("quick routine question:
since I was laid off, can I push my premium back?"). Signals match the underlying situation -
job loss, inability to pay, coercion, requests to skip controls - not specific wording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.logging_config import get_logger, trace

logger = get_logger(__name__)

_I = re.IGNORECASE
_PAYMENTS = r"(?:payments?|premiums?|bills?|repayments?|instal?ments?|rent|mortgage|loan|debits?|direct\s+debits?)"
_RELATIVE = (
    r"(?:carer|caregiver|son|daughter|nephew|niece|grandson|granddaughter|partner|husband|wife|boyfriend|"
    r"girlfriend|relative|family\s+member|brother|sister|neighbou?r|friend|ex)"
)

SIGNALS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "safeguarding_concern": [
        ("explicit_safeguarding", re.compile(r"\bsafeguarding\b", _I)),
        ("abuse", re.compile(r"\b(?:abuse[ds]?|abusive|domestic\s+violence|elder\s+abuse|financial\s+abuse)\b", _I)),
        ("coercion", re.compile(r"\b(?:coerc\w*|forc(?:ed|ing)\s+me|pressur(?:ed|ing)\s+me|making\s+me|blackmail\w*|extort\w*)\b", _I)),
        ("fear", re.compile(r"\b(?:threaten\w*|i(?:'m|\s+am)\s+(?:scared|afraid|frightened)|scared\s+(?:of|to)|afraid\s+(?:of|to))\b", _I)),
        ("third_party_control", re.compile(
            rf"\b(?:my\s+)?{_RELATIVE}\s+(?:keeps|is|has\s+been|have\s+been|always)\s+(?:taking|using|withdrawing|"
            r"demanding|controlling|moving|spending|asking\s+(?:me\s+)?for)\b", _I)),
        ("unauthorised_use", re.compile(
            r"\bsomeone\s+(?:else\s+)?(?:is\s+|has\s+been\s+|keeps\s+)?(?:using|accessing|taking|withdrawing|draining|controlling)\s+"
            r"(?:money\s+from\s+)?(?:my|our)\s+(?:money|account|card|savings|pension)\b", _I)),
        ("instructed_transfer", re.compile(
            r"\b(?:told|instructed|asked|telling)\s+me\s+to\s+(?:urgently\s+)?(?:move|transfer|withdraw|send)\s+(?:all\s+)?(?:of\s+)?(?:my\s+)?"
            r"(?:money|savings|funds|pension)\b", _I)),
        ("vulnerability", re.compile(r"\b(?:vulnerable|dementia|self[-\s]?harm|suicid\w*|hurt\s+myself|don'?t\s+want\s+to\s+live)\b", _I)),
    ],
    "financial_hardship": [
        ("explicit_hardship", re.compile(r"\b(?:financial\s+)?hardship\b", _I)),
        ("job_loss", re.compile(
            r"\b(?:lost\s+my\s+(?:job|income|work)|laid\s+off|made\s+redundant|redundan(?:t|cy)|out\s+of\s+work|unemployed|"
            r"between\s+jobs|no\s+(?:longer\s+(?:have\s+)?(?:an?\s+)?)?income|until\s+i\s+(?:find|get)\s+(?:a\s+)?(?:new\s+)?(?:job|work))\b", _I)),
        ("cannot_afford", re.compile(
            rf"\b(?:can'?t|cannot|can\s+not|couldn'?t|unable\s+to|struggl\w*\s+to|no\s+way\s+to)\s+(?:afford|pay|make|meet|keep\s+up\s+with|cover)\b"
            rf"(?:.{{0,30}}\b{_PAYMENTS}\b)?", _I)),
        ("struggling_with_payments", re.compile(rf"\bstruggl\w*\s+(?:with|to\s+pay|to\s+make)\b.{{0,30}}\b{_PAYMENTS}\b", _I)),
        ("arrears", re.compile(rf"\b(?:behind\s+on|fall(?:en|ing)?\s+behind(?:\s+on)?|missed\s+(?:several|multiple|a\s+few|two|three)|in\s+arrears)\b", _I)),
        ("money_tight", re.compile(r"\b(?:money\s+is\s+(?:really\s+|very\s+|so\s+)?tight|tight\s+on\s+(?:money|cash)|short\s+on\s+(?:money|cash|funds)|broke)\b", _I)),
        ("insolvency", re.compile(r"\b(?:bankrupt\w*|insolven\w*|foreclos\w*|repossess\w*|evict\w*|debt\s+collector\w*|payday\s+loan\w*)\b", _I)),
        ("payment_relief", re.compile(r"\b(?:payment\s+holiday|hardship\s+(?:plan|program(?:me)?)|debt\s+relief)\b", _I)),
    ],
    "compliance_override_request": [
        ("skip_controls", re.compile(
            r"\b(?:bypass|skip|waive|ignore|override|get\s+(?:a)?round|circumvent|avoid|sidestep|turn\s+off|disable|lift|remove)\s+"
            r"(?:the\s+|my\s+|this\s+|that\s+|these\s+|those\s+|all\s+|any\s+)?(?:\w+\s+)?"
            r"(?:kyc|know[-\s]your[-\s]customer|verification|identity\s+checks?|id\s+checks?|aml|anti[-\s]money|sanctions?|"
            r"compliance|checks?|restrictions?|controls?|freeze|hold|due\s+diligence|limits?)\b", _I)),
        ("without_checks", re.compile(
            r"\bwithout\s+(?:any\s+|the\s+)?(?:kyc|verification|verifying|id|identity\s+checks?|checks|paperwork|documents?|approval)\b", _I)),
        ("exception_request", re.compile(
            r"\b(?:just\s+this\s+once|make\s+an\s+exception|as\s+an\s+exception|special\s+exception|look\s+the\s+other\s+way|"
            r"off\s+the\s+record|don'?t\s+(?:report|flag|record|log)\s+(?:it|this|that)|keep\s+(?:it|this)\s+off\s+the\s+(?:books|record))\b", _I)),
        ("despite_restriction", re.compile(
            r"\b(?:process|approve|set\s+(?:it\s+)?up|allow|push|run|do)\b.{0,60}\banyway\b|"
            r"\beven\s+though\b.{0,40}\b(?:restricted|frozen|on\s+hold|not\s+verified|blocked)\b", _I)),
        ("structuring", re.compile(
            r"\b(?:split|break)\b.{0,40}\b(?:smaller|multiple|several)\b.{0,40}\b(?:so|to\s+avoid)\b.{0,30}\b(?:flag\w*|report\w*|notic\w*|detect\w*)\b", _I)),
        ("backdating", re.compile(r"\bback-?dat\w*\b", _I)),
    ],
}

# When several categories are present the most protective one is primary.
PRECEDENCE = ("safeguarding_concern", "financial_hardship", "compliance_override_request")


@dataclass(frozen=True)
class SignalResult:
    category: str  # "none" or one of PRECEDENCE
    signals: tuple[str, ...]  # e.g. ("financial_hardship:job_loss",)


@trace(logger)
def detect_hard_category(text: str) -> SignalResult:
    """Return the primary hard category found in ``text`` and every signal that matched."""
    if not isinstance(text, str) or not text:
        return SignalResult("none", ())
    normalized = re.sub(r"[’‘`]", "'", text)
    found: list[str] = []
    categories: list[str] = []
    for category in PRECEDENCE:
        for name, pattern in SIGNALS[category]:
            if pattern.search(normalized):
                found.append(f"{category}:{name}")
                if category not in categories:
                    categories.append(category)
    return SignalResult(categories[0] if categories else "none", tuple(found))


@trace(logger)
def strongest_category(*categories: str) -> str:
    """Combine category opinions: any hard category wins over ``none``; ties use PRECEDENCE."""
    hard = [c for c in categories if c in PRECEDENCE]
    if not hard:
        return "none"
    return min(hard, key=PRECEDENCE.index)
