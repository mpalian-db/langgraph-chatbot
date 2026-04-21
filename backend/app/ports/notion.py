from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class NotionPage:
    id: str
    title: str
    text: str
    url: str


@runtime_checkable
class NotionPort(Protocol):
    async def list_pages(self, database_id: str) -> list[NotionPage]: ...

    async def get_page_content(self, page_id: str) -> NotionPage: ...
