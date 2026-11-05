"""Minimal Telegram Bot API client: send messages and read replies.

The bot token is a password: it is never printed. Errors are re-raised with
the token redacted, because GitHub Actions logs of a public repo are public.
"""
from __future__ import annotations

import requests

MAX_LEN = 4000   # Telegram's limit is 4096 characters per message


class TelegramError(RuntimeError):
    pass


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
    def __init__(self, token: str, chat_id: str | int | None, timeout: float = 20.0):
        if not token:
            raise TelegramError("TELEGRAM_BOT_TOKEN is not set.")
        self.token = token
        self.chat_id = str(chat_id) if chat_id is not None else None
        self.timeout = timeout

    def redact(self, s: str) -> str:
        return s.replace(self.token, "<token>")

    def _call(self, method: str, **params):
        try:
            r = requests.post(f"https://api.telegram.org/bot{self.token}/{method}",
                              json=params, timeout=self.timeout)
            js = r.json()
        except Exception as e:   # the exception text can contain the URL, i.e. the token
            raise TelegramError(self.redact(f"Telegram {method} failed: {type(e).__name__}: {e}")) from None
        if not js.get("ok"):
            raise TelegramError(self.redact(f"Telegram {method} error: {js.get('description', 'unknown')}"))
        return js["result"]

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
