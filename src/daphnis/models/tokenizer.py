"""Optional, digest-verified local text tokenizers."""

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

from ..token_budget import DEFAULT_COUNTER, TokenCounter

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def verified_tokenizer(path: Path, sha256: str) -> TokenCounter:
    try:
        from tokenizers import Tokenizer

        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            logger.warning("Local tokenizer digest mismatch; using the token estimate.")
            return DEFAULT_COUNTER
        tokenizer = Tokenizer.from_str(data.decode("utf-8"))
    except Exception:  # noqa: BLE001 - Optional native bindings can also reject a file.
        logger.warning("Local tokenizer unavailable; using the token estimate.")
        return DEFAULT_COUNTER

    def count(text: str) -> int:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False).ids)
        except UnicodeError:
            escaped = text.encode("utf-8", errors="backslashreplace").decode("utf-8")
            return len(tokenizer.encode(escaped, add_special_tokens=False).ids)

    return TokenCounter(count, "local_tokenizer")
