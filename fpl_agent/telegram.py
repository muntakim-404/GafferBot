"""Minimal Telegram Bot API client: send messages and read replies.

The bot token is a password: it is never printed. Errors are re-raised with
the token redacted, because GitHub Actions logs of a public repo are public.
"""
from __future__ import annotations

import time

import requests

MAX_LEN = 4000   # Telegram's limit is 4096 characters per message
PERMANENT_CODES = {400, 401, 403, 404}   # bad chat id, bad/revoked token, bot blocked: retrying won't help


class TelegramError(RuntimeError):
    def __init__(self, msg: str, permanent: bool = False):
        super().__init__(msg)
        self.permanent = permanent


def split_message(text: str, limit: int = MAX_LEN) -> list[str]:
    chunks, cur = [], ""
    for line in text.split("\n"):
        while len(line) > limit:                      # pathological single long line
            chunks.append(line[:limit])
            line = line[limit:]
        if len(cur) + len(line) + 1 > limit:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


class Telegram:
    def __init__(self, token: str, chat_id: str | int | None, timeout: float = 20.0,
                 retries: int = 3, backoff: float = 3.0):
        if not token:
            raise TelegramError("TELEGRAM_BOT_TOKEN is not set.", permanent=True)
        self.token = token
        self.chat_id = str(chat_id) if chat_id is not None else None
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def redact(self, s: str) -> str:
        return s.replace(self.token, "<token>")

    def _call(self, method: str, **params):
        """Retries network blips, 5xx and rate limits; fails at once on errors retrying can't fix."""
        last = f"Telegram {method} failed"
        for attempt in range(self.retries):
            if attempt:
                time.sleep(self.backoff * attempt)
            try:
                r = requests.post(f"https://api.telegram.org/bot{self.token}/{method}",
                                  json=params, timeout=self.timeout)
            except Exception as e:   # the exception text can contain the URL, i.e. the token
                last = self.redact(f"Telegram {method} failed: {type(e).__name__}: {e}")
                continue
            try:
                js = r.json()
            except ValueError:       # e.g. an HTML error page from a proxy
                js = {"ok": False, "error_code": r.status_code, "description": f"HTTP {r.status_code}"}
            if js.get("ok"):
                return js["result"]
            code = int(js.get("error_code") or r.status_code or 0)
            last = self.redact(f"Telegram {method} error {code}: {js.get('description', 'unknown')}")
            if code in PERMANENT_CODES:
                raise TelegramError(last, permanent=True)
            if code == 429:          # rate limited: Telegram says how long to wait
                time.sleep(min(float((js.get("parameters") or {}).get("retry_after", 1)), 30.0))
        raise TelegramError(last)

    def send(self, text: str, chat_id: str | None = None) -> None:
        target = chat_id or self.chat_id
        if not target:
            raise TelegramError("TELEGRAM_CHAT_ID is not set.")
        for chunk in split_message(text):
            self._call("sendMessage", chat_id=target, text=chunk, parse_mode="HTML",
                       disable_web_page_preview=True)

    def updates(self, offset: int) -> list[dict]:
        """Messages sent to the bot since `offset` (an update_id + 1)."""
        return self._call("getUpdates", offset=offset, timeout=0, allowed_updates=["message"])


class DryRunTelegram:
    """Prints instead of sending. Used by `tick --dry-run` for local testing."""

    def __init__(self, chat_id: str = "dry-run"):
        self.chat_id = chat_id
        self.sent: list[str] = []

    def send(self, text: str, chat_id: str | None = None) -> None:
        self.sent.append(text)
        print("\n----- Telegram message -----\n" + text + "\n----------------------------")

    def updates(self, offset: int) -> list[dict]:
        return []
