"""Read-only access to the local knowledge-base text files.

Used by the ``knowledgebase://{document_id}/excerpt`` Resource and by retrieval (product-document
parsing and explicit test-document injection). It never exposes a bulk corpus: callers get one
document's metadata or a bounded excerpt.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.logging_config import get_logger, trace

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_KB_ROOT = BASE_DIR / "knowledge_base" / "nexus_unstructured_knowledge_base"
DOCUMENT_ID_RE = re.compile(r"^[A-Z]{2,6}(?:-[A-Z0-9]{2,6}){1,3}(?:-[a-z0-9]+)+$")
_HEADER_KEYS = (
    "DOCUMENT_ID", "TITLE", "CATEGORY", "PRODUCT_LINE", "SUBTOPIC", "DOCUMENT_TYPE", "VERSION",
    "EFFECTIVE_DATE", "AUDIENCE", "AUTHORITY", "SOURCE_CLASS", "STRUCTURED_DATA_ROLE", "RETRIEVAL_KEYWORDS",
)
_HEADER_RE = re.compile(
    rf"\b({'|'.join(_HEADER_KEYS)}):\s*(.*?)(?=\s+\b(?:{'|'.join(_HEADER_KEYS)}):|\s+\bCONTENT:|$)", re.DOTALL
)


@dataclass(frozen=True)
class KBDocument:
    document_id: str
    path: Path
    relative_path: str
    title: str
    category: str
    product_line: str
    subtopic: str
    is_test: bool
    body: str


@trace(logger)
def kb_root() -> Path:
    return Path(os.getenv("NEXUS_KB_ROOT", str(DEFAULT_KB_ROOT)))


@trace(logger)
def parse_header(text: str) -> dict[str, str]:
    head = text.split("CONTENT:", 1)[0]
    return {key.lower(): re.sub(r"\s+", " ", value).strip() for key, value in _HEADER_RE.findall(head)}


@trace(logger)
def _load(path: Path, root: Path) -> KBDocument:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    header = parse_header(text)
    relative = path.relative_to(root).as_posix()
    body = text.split("CONTENT:", 1)[1].strip() if "CONTENT:" in text else text.strip()
    category = header.get("category", "").lower()
    return KBDocument(
        document_id=header.get("document_id") or path.stem,
        path=path,
        relative_path=relative,
        title=header.get("title", path.stem),
        category=category,
        product_line=header.get("product_line", "").lower(),
        subtopic=header.get("subtopic", "").lower(),
        is_test=category == "test" or relative.lower().startswith("faqs/test/"),
        body=body,
    )


@lru_cache(maxsize=4)
@trace(logger)
def _catalogue(root: str) -> dict[str, KBDocument]:
    base = Path(root)
    documents: dict[str, KBDocument] = {}
    if base.is_dir():
        for path in sorted(base.rglob("*.txt")):
            if path.name.lower() == "readme.txt":
                continue
            document = _load(path, base)
            documents[document.document_id] = document
    return documents


@trace(logger)
def get_document(document_id: str, *, include_test: bool = False) -> KBDocument | None:
    """Return one document by its DOCUMENT_ID, or ``None``. Test/adversarial documents are hidden
    unless ``include_test`` is set (explicit test mode only)."""
    if not isinstance(document_id, str) or not DOCUMENT_ID_RE.fullmatch(document_id):
        return None
    document = _catalogue(str(kb_root())).get(document_id)
    if document is None or (document.is_test and not include_test):
        return None
    return document


@trace(logger)
def documents_by_source_path() -> dict[str, KBDocument]:
    """Map lower-cased relative source paths (the index's ``source_path``) to documents."""
    return {doc.relative_path.lower(): doc for doc in _catalogue(str(kb_root())).values()}


@trace(logger)
def bounded_excerpt(document: KBDocument, max_chars: int = 1200) -> tuple[str, bool]:
    """First paragraphs of the document body, cut at a paragraph boundary within ``max_chars``."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document.body) if p.strip()]
    excerpt = ""
    for paragraph in paragraphs:
        candidate = f"{excerpt}\n\n{paragraph}" if excerpt else paragraph
        if len(candidate) > max_chars:
            break
        excerpt = candidate
    if not excerpt and paragraphs:
        excerpt = paragraphs[0][:max_chars]
    return excerpt, len(excerpt) < len(document.body)


@trace(logger)
def product_minimum_risk_profile(document: KBDocument) -> str | None:
    """Parse the 'Minimum risk profile' stated in a product document's alignment block."""
    match = re.search(r"Minimum\s+risk\s+profile:\s*(conservative|moderate|aggressive)", document.body, re.IGNORECASE)
    return match.group(1).lower() if match else None


@trace(logger)
def product_document_for(product_id: str) -> KBDocument | None:
    for document in _catalogue(str(kb_root())).values():
        if document.document_id.startswith(f"{product_id}-"):
            return document
    return None
