from __future__ import annotations
 
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.logging_config import get_logger, trace

logger = get_logger("build_index")
 
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - handled during runtime dependency checks.
    load_dotenv = None  # type: ignore[assignment]
 
 
INDEX_VERSION = "1.0.0"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "nexus_knowledge"
SUPPORTED_EXTENSIONS = {".md", ".txt"}
# Package documentation, not knowledge: never embedded or retrievable.
NON_KNOWLEDGE_FILES = {"readme.txt", "readme.md", "manifest.csv"}
BASE_DIR = Path(__file__).resolve().parent
KB_ROOT = BASE_DIR / "knowledge_base"
VECTOR_ROOT = BASE_DIR / "vector_store"
ACTIVE_CHROMA_DIR = VECTOR_ROOT / "chroma"
MANIFEST_PATH = VECTOR_ROOT / "manifest.json"
CHUNK_TARGET_CHARS = 1800
CHUNK_OVERLAP_CHARS = 250
EMBED_BATCH_SIZE = 16
REQUIRED_METADATA_FIELDS = {
    "document_id",
    "product_line",
    "document_type",
    "topic",
    "document_hash",
    "chunk_id",
    "chunk_index",
    "section",
    "source_path",
    "index_version",
    "embedding_model",
}
HEADER_KEYS = (
    "DOCUMENT_ID",
    "TITLE",
    "CATEGORY",
    "PRODUCT_LINE",
    "SUBTOPIC",
    "DOCUMENT_TYPE",
    "VERSION",
    "EFFECTIVE_DATE",
    "AUDIENCE",
    "AUTHORITY",
    "SOURCE_CLASS",
    "STRUCTURED_DATA_ROLE",
    "RETRIEVAL_KEYWORDS",
)
 
 
QUIET = False  # the offline demo index reuses build_chunks() without progress output


class IndexBuildError(RuntimeError):
    """Fatal, actionable failure during offline index construction."""
 
 
@trace(logger)
def log(stage: str, message: str) -> None:
    if not QUIET:
        print(f"[{stage}] {message}", flush=True)
 
 
@trace(logger)
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
 
 
@trace(logger)
def normalize_relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix().lower()
 
 
@trace(logger)
def load_environment() -> None:
    from app.runtime import bootstrap

    bootstrap()  # repo-root .env + OS certificate store for the OpenAI connection
 
 
@trace(logger)
def require_runtime_dependencies() -> tuple[Any, Any, Any]:
    missing: list[str] = []
 
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        chromadb = None
        Settings = None
        missing.append("chromadb")
 
    try:
        from openai import OpenAI
    except ImportError:
        OpenAI = None
        missing.append("openai")
 
    if missing:
        raise IndexBuildError(
            "Missing runtime dependency/dependencies: "
            + ", ".join(missing)
            + ". Add/install them from requirements.txt before running the indexer."
        )
 
    return chromadb, Settings, OpenAI
 
 
@trace(logger)
def discover_documents(kb_root: Path) -> tuple[list[Path], list[Path]]:
    if not kb_root.exists():
        raise IndexBuildError(
            f"Knowledge-base directory not found: {kb_root.resolve()}"
        )
    if not kb_root.is_dir():
        raise IndexBuildError(
            f"Knowledge-base path is not a directory: {kb_root.resolve()}"
        )
 
    supported: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(kb_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.lower() in NON_KNOWLEDGE_FILES:
            skipped.append(path)
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            supported.append(path)
        else:
            skipped.append(path)
 
    if not supported:
        raise IndexBuildError(
            f"No supported documents found under {kb_root.resolve()}; "
            f"supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
 
    return supported, skipped
 
 
@trace(logger)
def load_document(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise IndexBuildError(f"Unable to read {path}: {exc}") from exc
 
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IndexBuildError(
            f"Invalid text encoding in {path}; expected UTF-8 compatible text: {exc}"
        ) from exc
 
 
@trace(logger)
def normalize_document(content: str) -> str:
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\bCONTENT:\s+(?=\d+\.\s+)", "CONTENT:\n\n", text)
    return text.strip()
 
 
@trace(logger)
def parse_header_metadata(content: str) -> dict[str, str]:
    key_pattern = "|".join(re.escape(key) for key in HEADER_KEYS)
    pattern = re.compile(
        rf"\b({key_pattern}):\s*(.*?)(?=\s+\b(?:{key_pattern}):|\s+\bCONTENT:|$)",
        re.DOTALL,
    )
    metadata: dict[str, str] = {}
    for key, value in pattern.findall(content):
        metadata[key.lower()] = re.sub(r"\s+", " ", value).strip()
    return metadata
 
 
@trace(logger)
def infer_document_type(parts: tuple[str, ...], header: dict[str, str]) -> str:
    if header.get("document_type"):
        return header["document_type"].lower()
    if header.get("category"):
        return header["category"].lower()
 
    top_level = next((part for part in parts if part != "nexus_unstructured_knowledge_base"), "")
    return {
        "policy_docs": "policy",
        "product_docs": "product",
        "faqs": "faq",
        "communication_templates": "template",
        "regulatory_notes": "regulatory",
    }.get(top_level, top_level or "general")
 
 
@trace(logger)
def infer_product_line(parts: tuple[str, ...], header: dict[str, str]) -> str:
    if header.get("product_line"):
        return header["product_line"].lower()
 
    product_lines = {"banking", "insurance", "wealth", "cross_product", "test"}
    for part in parts:
        normalized = part.lower()
        if normalized in product_lines:
            return normalized
    return "cross_product"
 
 
@trace(logger)
def infer_topic(path: Path, header: dict[str, str]) -> str:
    if header.get("subtopic"):
        return header["subtopic"].lower()
    return re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_") or "general"
 
 
@trace(logger)
def extract_metadata(path: Path, kb_root: Path, content: str) -> dict[str, str]:
    source_path = normalize_relative_path(path, kb_root)
    header = parse_header_metadata(content)
    parts = Path(source_path).parts
 
    return {
        "document_id": source_path,
        "document_hash": sha256_text(content),
        "source_path": source_path,
        "product_line": infer_product_line(parts, header),
        "document_type": infer_document_type(parts, header),
        "topic": infer_topic(path, header),
        "title": header.get("title") or path.stem.replace("-", " ").replace("_", " "),
        "source_document_id": header.get("document_id", "") or path.stem,
        "category": header.get("category", "").lower(),
        "authority": header.get("authority", ""),
        "source_class": header.get("source_class", ""),
    }
 
 
@trace(logger)
def split_sections(content: str) -> list[tuple[str, str]]:
    lines = content.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "document"
    current_lines: list[str] = []
 
    heading_pattern = re.compile(
        r"^(?:#{1,6}\s+.+|\d+\.\s{0,3}[A-Z][A-Z0-9 ,/&()'-]{2,})$"
    )
 
    for line in lines:
        stripped = line.strip()
        if heading_pattern.match(stripped):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = re.sub(r"\s+", " ", stripped.lstrip("#").strip())
            current_lines = [stripped]
        else:
            current_lines.append(line)
 
    if current_lines:
        sections.append((current_title, current_lines))
 
    return [
        (title, "\n".join(section_lines).strip())
        for title, section_lines in sections
        if "\n".join(section_lines).strip()
    ]
 
 
@trace(logger)
def paragraph_blocks(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if blocks:
        return blocks
    return [text.strip()] if text.strip() else []
 
 
@trace(logger)
def split_oversized_block(block: str, target_chars: int) -> list[str]:
    if len(block) <= target_chars:
        return [block]
 
    sentences = re.split(r"(?<=[.!?])\s+", block)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > target_chars:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
 
    if not parts:
        return [block[i : i + target_chars] for i in range(0, len(block), target_chars)]
    return parts
 
 
@trace(logger)
def chunk_section(section_title: str, section_text: str) -> list[str]:
    context_prefix = f"Section: {section_title}\n\n"
    target = CHUNK_TARGET_CHARS
    chunks: list[str] = []
    current = ""
 
    for block in paragraph_blocks(section_text):
        for part in split_oversized_block(block, target):
            candidate = f"{current}\n\n{part}".strip() if current else part
            if current and len(context_prefix) + len(candidate) > target:
                chunks.append((context_prefix + current).strip())
                overlap = current[-CHUNK_OVERLAP_CHARS:].strip()
                current = f"{overlap}\n\n{part}".strip() if overlap else part
            else:
                current = candidate
 
    if current:
        chunks.append((context_prefix + current).strip())
 
    return chunks
 
 
@trace(logger)
def chunk_document(document: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for section_title, section_text in split_sections(document["content"]):
        for chunk_text in chunk_section(section_title, section_text):
            chunk_index = len(chunks)
            metadata = {
                "document_id": document["metadata"]["document_id"],
                "product_line": document["metadata"]["product_line"],
                "document_type": document["metadata"]["document_type"],
                "topic": document["metadata"]["topic"],
                "document_hash": document["metadata"]["document_hash"],
                "chunk_index": chunk_index,
                "section": section_title,
                "source_path": document["metadata"]["source_path"],
                "index_version": INDEX_VERSION,
                "embedding_model": document["embedding_model"],
                "title": document["metadata"]["title"],
                "source_document_id": document["metadata"]["source_document_id"],
                "category": document["metadata"]["category"],
                "authority": document["metadata"]["authority"],
                "source_class": document["metadata"]["source_class"],
            }
            chunk_hash = sha256_text(normalize_chunk_for_hash(chunk_text))
            chunk_id = (
                f"{metadata['document_id']}:chunk-{chunk_index:04d}:"
                f"{chunk_hash[:16]}"
            )
            metadata["chunk_id"] = chunk_id
            metadata["chunk_hash"] = chunk_hash
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": metadata,
                    "chunk_hash": chunk_hash,
                }
            )
    return chunks
 
 
@trace(logger)
def normalize_chunk_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
 
 
@trace(logger)
def is_meaningful_chunk(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    alphanumeric_count = sum(char.isalnum() for char in normalized)
    if alphanumeric_count < 20 and len(normalized) < 80:
        return False
    if "\ufffd" in normalized:
        return False
    return True
 
 
@trace(logger)
def validate_chunks(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    accepted: list[dict[str, Any]] = []
    rejected = 0
    for chunk in chunks:
        if not is_meaningful_chunk(chunk["text"]):
            rejected += 1
            continue
        missing = REQUIRED_METADATA_FIELDS - set(chunk["metadata"])
        if missing:
            raise IndexBuildError(
                f"Chunk {chunk.get('id', '<unknown>')} is missing metadata fields: "
                + ", ".join(sorted(missing))
            )
        accepted.append(chunk)
    return accepted, rejected
 
 
@trace(logger)
def deduplicate_chunks(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    retained: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for chunk in chunks:
        chunk_hash = chunk["chunk_hash"]
        if chunk_hash in seen_hashes:
            continue
        seen_hashes.add(chunk_hash)
        retained.append(chunk)
    return retained, len(chunks) - len(retained)
 
 
@trace(logger)
def load_documents(paths: list[Path], kb_root: Path, embedding_model: str) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in paths:
        raw = load_document(path)
        normalized = normalize_document(raw)
        if not normalized:
            raise IndexBuildError(f"Document is empty after normalization: {path}")
        documents.append(
            {
                "path": path,
                "content": normalized,
                "metadata": extract_metadata(path, kb_root, normalized),
                "embedding_model": embedding_model,
            }
        )
    return documents
 
 
@trace(logger)
def transient_openai_errors(openai_module: Any) -> tuple[type[BaseException], ...]:
    names = (
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    )
    return tuple(
        error_type
        for name in names
        if isinstance((error_type := getattr(openai_module, name, None)), type)
    )
 
 
@trace(logger)
def detect_failed_embedding_response(exc: BaseException) -> bool:
    text = str(exc).lower()
    html_tokens = (
        "<!doctype html",
        "<html",
        "zscaler",
        "access denied",
        "blocked",
        "proxy",
        "security",
        "firewall",
        "gateway",
        "web filter",
    )
    return any(token in text for token in html_tokens)
 
 
@trace(logger)
def generate_embeddings(
    chunks: list[dict[str, Any]],
    embedding_model: str,
    OpenAI: Any,
) -> tuple[list[list[float]], int]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise IndexBuildError(
            "OPENAI_API_KEY is required for embedding generation and is not set."
        )
 
    import openai as openai_module
 
    client = OpenAI(api_key=api_key)
    transient_errors = transient_openai_errors(openai_module)
    expected_dimension: int | None = None
 
    @trace(logger)
    def embed_batch(batch: list[dict[str, Any]], start: int) -> list[list[float]]:
        nonlocal expected_dimension
        inputs = [chunk["text"] for chunk in batch]
 
        for attempt in range(1, 5):
            try:
                response = client.embeddings.create(
                    model=embedding_model,
                    input=inputs,
                )
                response_items = sorted(
                    response.data,
                    key=lambda item: getattr(item, "index", 0),
                )
                batch_embeddings = [item.embedding for item in response_items]
                if len(batch_embeddings) != len(batch):
                    raise IndexBuildError(
                        "OpenAI embedding response count did not match input "
                        f"batch size at offset {start}."
                    )
                for vector in batch_embeddings:
                    dimension = len(vector)
                    if expected_dimension is None:
                        expected_dimension = dimension
                    elif dimension != expected_dimension:
                        raise IndexBuildError(
                            "Inconsistent embedding dimensions returned by OpenAI: "
                            f"expected {expected_dimension}, got {dimension}."
                        )
                return batch_embeddings
            except transient_errors as exc:  # type: ignore[misc]
                if detect_failed_embedding_response(exc):
                    break
                if attempt == 4:
                    raise IndexBuildError(
                        "OpenAI embedding request failed after retries at batch "
                        f"offset {start}: {exc}"
                    ) from exc
                time.sleep(2**attempt)
            except IndexBuildError:
                raise
            except Exception as exc:
                if detect_failed_embedding_response(exc):
                    break
                raise IndexBuildError(
                    f"OpenAI embedding request failed at batch offset {start}: {exc}"
                ) from exc
        else:  # pragma: no cover - the loop always returns, raises, or breaks.
            raise IndexBuildError(f"Embedding request failed at batch offset {start}.")
 
        if len(batch) > 1:
            midpoint = len(batch) // 2
            log(
                "EMBED",
                f"Network filter rejected batch at offset {start}; "
                f"retrying as {midpoint} and {len(batch) - midpoint} chunks",
            )
            return embed_batch(batch[:midpoint], start) + embed_batch(
                batch[midpoint:], start + midpoint
            )
 
        metadata = batch[0]["metadata"]
        raise IndexBuildError(
            "Zscaler rejected an individual knowledge-base chunk, so changing the "
            "OpenAI client or embedding model will not resolve it. Blocked source: "
            f"{metadata['source_path']} (chunk {metadata['chunk_index']}). "
            "Use an organization-approved OpenAI/Azure endpoint or request a DLP "
            "exception for this project."
        )
 
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        embeddings.extend(embed_batch(chunks[start : start + EMBED_BATCH_SIZE], start))
 
    if len(embeddings) != len(chunks):
        raise IndexBuildError(
            f"OpenAI embedding generation produced {len(embeddings)} vectors for "
            f"{len(chunks)} chunks. This usually indicates partial/failed batches."
        )
    return embeddings, 0
 
 
@trace(logger)
def chroma_client(chromadb: Any, Settings: Any, path: Path) -> Any:
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )
 
 
@trace(logger)
def close_chroma_client(client: Any) -> None:
    """Release Chroma's shared SQLite system so Windows can move the index."""
    try:
        system = getattr(client, "_system", None)
        if system is not None:
            system.stop()
    finally:
        try:
            from chromadb.api.client import SharedSystemClient
 
            SharedSystemClient.clear_system_cache()
        except (ImportError, AttributeError):
            pass
        gc.collect()
 
 
@trace(logger)
def create_chroma_collection(
    chromadb: Any,
    Settings: Any,
    chroma_path: Path,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> None:
    if len(chunks) != len(embeddings):
        raise IndexBuildError(
            f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})."
        )
 
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chroma_client(chromadb, Settings, chroma_path)
    try:
        existing_names = [
            collection if isinstance(collection, str) else collection.name
            for collection in client.list_collections()
        ]
        if COLLECTION_NAME in existing_names:
            client.delete_collection(COLLECTION_NAME)
 
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "index_version": INDEX_VERSION,
                "embedding_model": chunks[0]["metadata"]["embedding_model"] if chunks else "",
            },
            embedding_function=None,
        )
 
        collection.add(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks],
            embeddings=embeddings,
        )
    except Exception as exc:
        raise IndexBuildError(f"Failed to create Chroma collection: {exc}") from exc
    finally:
        if "collection" in locals():
            del collection
        close_chroma_client(client)
        del client
 
 
@trace(logger)
def build_manifest(
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    duplicate_count: int,
    failed_count: int,
    embedding_model: str,
    embedding_dimension: int,
) -> dict[str, Any]:
    document_hashes = {
        document["metadata"]["document_id"]: document["metadata"]["document_hash"]
        for document in documents
    }
    return {
        "index_version": INDEX_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vector_database": "chromadb",
        "collection_name": COLLECTION_NAME,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "duplicate_count": duplicate_count,
        "failed_count": failed_count,
        "knowledge_base_root": KB_ROOT.relative_to(BASE_DIR).as_posix(),
        "chroma_path": ACTIVE_CHROMA_DIR.relative_to(BASE_DIR).as_posix(),
        "document_hashes": document_hashes,
    }
 
 
@trace(logger)
def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
 
 
@trace(logger)
def validate_manifest(manifest: dict[str, Any], expected_chunk_count: int) -> None:
    expected = {
        "index_version": INDEX_VERSION,
        "vector_database": "chromadb",
        "collection_name": COLLECTION_NAME,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise IndexBuildError(
                f"Manifest mismatch for {key}: expected {value}, got {manifest.get(key)}"
            )
    if manifest.get("chunk_count") != expected_chunk_count:
        raise IndexBuildError(
            "Manifest chunk_count does not match retained chunk count: "
            f"{manifest.get('chunk_count')} != {expected_chunk_count}"
        )
 
 
@trace(logger)
def validate_chroma_index(
    chromadb: Any,
    Settings: Any,
    chroma_path: Path,
    manifest: dict[str, Any],
) -> None:
    validate_manifest(manifest, int(manifest["chunk_count"]))
    try:
        client = chroma_client(chromadb, Settings, chroma_path)
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
        count = collection.count()
        if count <= 0:
            raise IndexBuildError("Chroma collection contains no documents.")
        if count != manifest["chunk_count"]:
            raise IndexBuildError(
                f"Chroma count mismatch: collection has {count}, "
                f"manifest expects {manifest['chunk_count']}."
            )
 
        records = collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=count,
        )
    except IndexBuildError:
        raise
    except Exception as exc:
        raise IndexBuildError(f"Failed to reopen/validate Chroma collection: {exc}") from exc
    finally:
        if "client" in locals():
            if "collection" in locals():
                del collection
            close_chroma_client(client)
            del client
 
    ids = records.get("ids") or []
    documents = records.get("documents") or []
    metadatas = records.get("metadatas") or []
    embeddings = records.get("embeddings")
 
    if len(ids) != len(set(ids)):
        raise IndexBuildError("Chroma validation failed: duplicate stored IDs found.")
    if len(ids) != count or len(documents) != count or len(metadatas) != count:
        raise IndexBuildError("Chroma validation failed: stored record counts do not match.")
    if embeddings is None or len(embeddings) != count:
        raise IndexBuildError("Chroma validation failed: embeddings are missing or incomplete.")
 
    dimensions = {len(vector) for vector in embeddings}
    if len(dimensions) != 1:
        raise IndexBuildError(
            f"Chroma validation failed: inconsistent embedding dimensions {dimensions}."
        )
    dimension = dimensions.pop()
    if dimension != manifest["embedding_dimension"]:
        raise IndexBuildError(
            f"Manifest embedding_dimension {manifest['embedding_dimension']} does not "
            f"match stored vectors ({dimension})."
        )
 
    for record_id, document, metadata in zip(ids, documents, metadatas):
        if not isinstance(document, str) or not document.strip():
            raise IndexBuildError(f"Stored chunk is empty for ID {record_id}.")
        missing = REQUIRED_METADATA_FIELDS - set(metadata or {})
        if missing:
            raise IndexBuildError(
                f"Stored metadata for {record_id} is missing fields: "
                + ", ".join(sorted(missing))
            )
        if metadata.get("chunk_id") != record_id:
            raise IndexBuildError(
                f"Stored metadata chunk_id does not match Chroma ID for {record_id}."
            )
        if metadata.get("embedding_model") != manifest["embedding_model"]:
            raise IndexBuildError(
                f"Embedding model metadata mismatch for {record_id}: "
                f"{metadata.get('embedding_model')}."
            )
 
 
@trace(logger)
def unique_build_dir(vector_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    base = vector_root / f".build_{timestamp}_{os.getpid()}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = vector_root / f"{base.name}_{suffix}"
        suffix += 1
    return candidate
 
 
@trace(logger)
def activate_index(temp_chroma_dir: Path, temp_manifest_path: Path) -> None:
    VECTOR_ROOT.mkdir(parents=True, exist_ok=True)
    backup_dir = VECTOR_ROOT / f".previous_chroma_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    old_manifest_backup = VECTOR_ROOT / ".previous_manifest.json"
 
    try:
        if ACTIVE_CHROMA_DIR.exists():
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            ACTIVE_CHROMA_DIR.replace(backup_dir)
        if MANIFEST_PATH.exists():
            if old_manifest_backup.exists():
                old_manifest_backup.unlink()
            MANIFEST_PATH.replace(old_manifest_backup)
 
        temp_chroma_dir.replace(ACTIVE_CHROMA_DIR)
        shutil.copy2(temp_manifest_path, MANIFEST_PATH)
 
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if old_manifest_backup.exists():
            old_manifest_backup.unlink()
    except Exception as exc:
        if ACTIVE_CHROMA_DIR.exists():
            shutil.rmtree(ACTIVE_CHROMA_DIR, ignore_errors=True)
        if backup_dir.exists():
            backup_dir.replace(ACTIVE_CHROMA_DIR)
        if MANIFEST_PATH.exists():
            MANIFEST_PATH.unlink()
        if old_manifest_backup.exists():
            old_manifest_backup.replace(MANIFEST_PATH)
        raise IndexBuildError(
            f"Failed to activate new Chroma index; previous index was restored if present: {exc}"
        ) from exc
 
 
@trace(logger)
def build_chunks(
    documents_paths: list[Path] | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    kb_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Discover, normalise, chunk, validate and de-duplicate the knowledge base (no embedding).

    Shared by ``build_index`` and the offline demo/test index so both see identical chunks.
    """
    kb_root = kb_root or KB_ROOT
    if documents_paths is None:
        documents_paths, _ = discover_documents(kb_root)
    documents = load_documents(documents_paths, kb_root, embedding_model)
    log("INGEST", f"Loaded {len(documents)} documents")
    log("NORMALIZE", f"Normalized {len(documents)} documents")

    all_chunks: list[dict[str, Any]] = []
    for document in documents:
        all_chunks.extend(chunk_document(document))
    log("CHUNK", f"Created {len(all_chunks)} chunks")

    valid_chunks, rejected_count = validate_chunks(all_chunks)
    log("VALIDATE", f"Accepted {len(valid_chunks)} chunks")
    if rejected_count:
        log("VALIDATE", f"Rejected {rejected_count} invalid chunks")

    deduped_chunks, duplicate_count = deduplicate_chunks(valid_chunks)
    log("DEDUP", f"Removed {duplicate_count} duplicate chunks")
    if not deduped_chunks:
        raise IndexBuildError("No valid chunks remain after validation and deduplication.")
    return documents, deduped_chunks, duplicate_count


@trace(logger)
def build_index() -> None:
    log("INDEX", "Starting Nexus knowledge-base indexing")
    load_environment()
    chromadb, Settings, OpenAI = require_runtime_dependencies()
 
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    if embedding_model != DEFAULT_EMBEDDING_MODEL:
        log(
            "CONFIG",
            f"OPENAI_EMBEDDING_MODEL={embedding_model} "
            f"(default is {DEFAULT_EMBEDDING_MODEL})",
        )
 
    documents_paths, skipped = discover_documents(KB_ROOT)
    log("DISCOVERY", f"Found {len(documents_paths)} supported documents")
    if skipped:
        log("DISCOVERY", f"Skipped {len(skipped)} unsupported files")
 
    documents, deduped_chunks, duplicate_count = build_chunks(documents_paths, embedding_model)
 
    log("EMBED", f"Model: {embedding_model}")
    embeddings, failed_count = generate_embeddings(deduped_chunks, embedding_model, OpenAI)
    embedding_dimension = len(embeddings[0]) if embeddings else 0
    if embedding_dimension <= 0:
        raise IndexBuildError("No embeddings were generated.")
    log("EMBED", f"Generated {len(embeddings)} embeddings")
    log("EMBED", f"Dimension: {embedding_dimension}")
 
    temp_root = unique_build_dir(VECTOR_ROOT)
    temp_chroma_dir = temp_root / "chroma"
    temp_manifest_path = temp_root / "manifest.json"
    try:
        manifest = build_manifest(
            documents=documents,
            chunks=deduped_chunks,
            duplicate_count=duplicate_count,
            failed_count=failed_count,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )
        manifest["chroma_path"] = temp_chroma_dir.relative_to(BASE_DIR).as_posix()
 
        create_chroma_collection(
            chromadb=chromadb,
            Settings=Settings,
            chroma_path=temp_chroma_dir,
            chunks=deduped_chunks,
            embeddings=embeddings,
        )
        write_manifest(temp_manifest_path, manifest)
        validate_chroma_index(chromadb, Settings, temp_chroma_dir, manifest)
        log("VALIDATE", "Temporary Chroma index passed consistency checks")
 
        manifest["chroma_path"] = ACTIVE_CHROMA_DIR.relative_to(BASE_DIR).as_posix()
        write_manifest(temp_manifest_path, manifest)
        activate_index(temp_chroma_dir, temp_manifest_path)
        validate_chroma_index(chromadb, Settings, ACTIVE_CHROMA_DIR, manifest)
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
 
    log("CHROMA", f"Collection: {COLLECTION_NAME}")
    log("CHROMA", f"Persisted to {ACTIVE_CHROMA_DIR.as_posix()}/")
    log("VALIDATE", "Chroma index passed consistency checks")
    log("MANIFEST", f"Written to {MANIFEST_PATH.as_posix()}")
    log("INDEX", f"Final document count: {len(documents)}")
    log("INDEX", f"Final chunk/vector count: {len(deduped_chunks)}")
    log("INDEX", f"Duplicate chunks removed: {duplicate_count}")
    log("INDEX", "Indexing completed successfully")
 
 
@trace(logger)
def main() -> int:
    try:
        build_index()
    except IndexBuildError as exc:
        log("ERROR", str(exc))
        return 1
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())