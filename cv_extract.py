"""Extract plain text from résumé uploads (UTF-8 text or PDF)."""

import io
from typing import Final

_PDF_MAGIC: Final[bytes] = b"%PDF"


def extract_text_from_cv_bytes(raw: bytes) -> str:
    if not raw:
        return ""
    head = raw[: min(8, len(raw))]
    if raw.startswith(_PDF_MAGIC) or (head.startswith(b"%PDF")):
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            parts: list[str] = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            text = "\n".join(parts).strip()
            if text:
                return text
        except Exception:
            pass
    return raw.decode("utf-8", errors="ignore")
