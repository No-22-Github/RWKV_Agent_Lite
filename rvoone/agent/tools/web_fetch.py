from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from rvoone.agent.tools.base import Tool


def _strip_tags(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _validate_url(url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False, "Only http/https URLs are supported."
        if not parsed.netloc:
            return False, "Missing hostname."
        return True, ""
    except Exception as exc:
        return False, str(exc)


class WebFetchTool(Tool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch a URL and return its readable text."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch."},
                "maxChars": {"type": "integer", "minimum": 100},
            },
            "required": ["url"],
        }

    async def execute(self, **kwargs: Any) -> str:
        url = str(kwargs.get("url", "")).strip()
        max_chars = int(kwargs.get("maxChars", 50000))
        ok, error_msg = _validate_url(url)
        if not ok:
            return json.dumps({"error": error_msg, "url": url}, ensure_ascii=False)

        try:
            req = Request(url, headers={"User-Agent": "rvoone-web-fetch"})
            with urlopen(req, timeout=20) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                raw = resp.read()
                text = raw.decode(charset, errors="replace")
                content_type = resp.headers.get("content-type", "")
        except Exception as exc:
            return json.dumps({"error": str(exc), "url": url}, ensure_ascii=False)

        if "text/html" in content_type or text.lstrip().lower().startswith("<html"):
            text = _strip_tags(text)
        if len(text) > max_chars:
            text = text[:max_chars]
        return json.dumps(
            {"url": url, "content_type": content_type, "length": len(text), "text": text},
            ensure_ascii=False,
        )
