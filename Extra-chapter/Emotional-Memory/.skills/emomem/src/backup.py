"""emomem 裁剪版 backup（vendored）。

只保留 ``_IncrementalJSONReader``：一个 stdlib-only 的增量 JSON 游标，
被 ``src.memory.manager.restore_from_file``（懒导入）用于流式恢复备份。
上游完整版 src/backup.py 还依赖 local_store 与大量 settings 常量，此处
裁剪为自包含副本；EMOMEM 的 CLI 导入走 ``import_all``（json.load），
本文件的唯一消费者是 manager.restore_from_file。
"""

from __future__ import annotations

import json
from typing import Any


class _IncrementalJSONReader:
    """Small stdlib-only JSON cursor that bounds memory to one decoded value."""

    CHUNK_SIZE = 64 * 1024
    MAX_BUFFER_BYTES = 100 * 1024 * 1024  # single JSON value ceiling

    def __init__(self, stream) -> None:
        self.stream = stream
        self.decoder = json.JSONDecoder()
        self.buffer = ""
        self.position = 0
        self.eof = False
        self._buffer_bytes = 0

    def _fill(self) -> bool:
        chunk = self.stream.read(self.CHUNK_SIZE)
        if chunk:
            self.buffer += chunk
            self._buffer_bytes += len(chunk.encode("utf-8"))
            if self._buffer_bytes > self.MAX_BUFFER_BYTES:
                raise ValueError("Backup JSON value exceeds the safe size limit")
            return True
        self.eof = True
        return False

    def _compact(self) -> None:
        if self.position >= self.CHUNK_SIZE:
            dropped = self.buffer[: self.position]
            self.buffer = self.buffer[self.position :]
            self._buffer_bytes -= len(dropped.encode("utf-8"))
            self.position = 0

    def _skip_space(self) -> None:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer) or self.eof:
                self._compact()
                return
            self._fill()

    def peek(self) -> str:
        self._skip_space()
        if self.position >= len(self.buffer):
            raise ValueError("Backup JSON ended unexpectedly")
        return self.buffer[self.position]

    def expect(self, character: str) -> None:
        if self.peek() != character:
            raise ValueError(f"Malformed backup JSON: expected {character!r}")
        self.position += 1
        self._compact()

    def value(self) -> Any:
        self._skip_space()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as exc:
                if self.eof:
                    raise ValueError("Backup JSON is malformed or truncated") from exc
                self._fill()
                continue
            self.position = end
            self._compact()
            return value

    def ensure_finished(self) -> None:
        self._skip_space()
        if self.position < len(self.buffer):
            raise ValueError("Backup JSON has trailing data")
        if not self.eof:
            self._fill()
            self._skip_space()
            if self.position < len(self.buffer):
                raise ValueError("Backup JSON has trailing data")