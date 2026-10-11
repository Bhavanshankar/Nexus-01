"""Process bootstrap shared by every entry point (orchestrator, index builder, retrieval CLI).

* Loads ``.env`` from the repository root (never from the working directory of a caller).
* Uses the operating-system certificate store for outbound TLS via ``truststore``. Corporate
  networks often re-sign TLS with an internal CA that is trusted by Windows/macOS but not by
  ``certifi``; using the OS store keeps verification ON instead of disabling it. Set
  ``NEXUS_USE_SYSTEM_TRUSTSTORE=0`` to opt out.
* Disables third-party telemetry (CrewAI/OpenTelemetry) so customer data never leaves the process
  through analytics.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.logging_config import get_logger, trace

logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
_BOOTSTRAPPED = False


@trace(logger)
def bootstrap() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(BASE_DIR / ".env", override=False)
    except ImportError:
        pass
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    if os.getenv("NEXUS_USE_SYSTEM_TRUSTSTORE", "1") != "0":
        try:
            import truststore

            truststore.inject_into_ssl()
        except ImportError:
            logger.warning("truststore is not installed; using the default certificate bundle")
    _BOOTSTRAPPED = True
