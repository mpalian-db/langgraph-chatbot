from __future__ import annotations

import pathlib

import aiofiles
import aiofiles.os


class LocalFileStorageAdapter:
    """Storage adapter backed by the local filesystem."""

    def __init__(self, base_dir: pathlib.Path) -> None:
        self._base = pathlib.Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    async def store(self, filename: str, content: bytes) -> str:
        """Store content to a file and return its absolute path."""
        path = self._base / filename
        async with aiofiles.open(path, "wb") as f:
            await f.write(content)
        return str(path)

    async def retrieve(self, path: str) -> bytes:
        """Retrieve content from a file."""
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        async with aiofiles.open(p, "rb") as f:
            return await f.read()

    async def delete(self, path: str) -> None:
        """Delete a file."""
        await aiofiles.os.remove(path)
