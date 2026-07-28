"""Deterministic HMAC-based identifiers for posts, pages, speeches and paragraphs."""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path


def normalize_raw_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>"}:
        return ""
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text[:-2]
    return text


def load_or_create_salt(temp_dir: str | Path) -> str:
    env_salt = os.environ.get("ANONYMIZATION_SALT")
    if env_salt:
        return env_salt

    salt_path = Path(temp_dir) / ".anonymization_salt"
    if salt_path.exists():
        salt = salt_path.read_text(encoding="utf-8").strip()
        if salt:
            return salt

    salt_path.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(32)
    salt_path.write_text(salt + "\n", encoding="utf-8")
    return salt


@dataclass(frozen=True)
class Anonymizer:
    salt: str
    digest_chars: int = 16

    def identifier(self, namespace: str, raw_value: object) -> str:
        raw = normalize_raw_id(raw_value)
        payload = f"{namespace}:{raw}".encode("utf-8")
        digest = hmac.new(self.salt.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        return f"{namespace}_{digest[: self.digest_chars]}"
