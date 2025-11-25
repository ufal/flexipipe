from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .doc import Document


@dataclass
class InputEntry:
    name: str
    aliases: tuple[str, ...]
    loader: Callable[..., Document]
    supports_stdin: bool = False

    def matches(self, value: str) -> bool:
        normalized = value.lower()
        return normalized == self.name.lower() or normalized in self.aliases

    def load(self, *, args, stdin_content: Optional[str] = None) -> Document:
        return self.loader(args=args, stdin_content=stdin_content)


@dataclass
class OutputEntry:
    name: str
    aliases: tuple[str, ...]
    saver: Callable[..., None]
    description: str = ""

    def matches(self, value: str) -> bool:
        normalized = value.lower()
        return normalized == self.name.lower() or normalized in self.aliases

    def save(self, document: Document, *, args, output_path: Optional[str], **kwargs) -> None:
        self.saver(document=document, args=args, output_path=output_path, **kwargs)


class IORegistry:
    def __init__(self) -> None:
        self._inputs: Dict[str, InputEntry] = {}
        self._outputs: Dict[str, OutputEntry] = {}

    def register_input(self, entry: InputEntry) -> None:
        key = entry.name.lower()
        self._inputs[key] = entry
        for alias in entry.aliases:
            self._inputs[alias.lower()] = entry

    def register_output(self, entry: OutputEntry) -> None:
        key = entry.name.lower()
        self._outputs[key] = entry
        for alias in entry.aliases:
            self._outputs[alias.lower()] = entry

    def get_input(self, name: str) -> Optional[InputEntry]:
        if not name:
            return None
        return self._inputs.get(name.lower())

    def get_output(self, name: str) -> Optional[OutputEntry]:
        if not name:
            return None
        return self._outputs.get(name.lower())


registry = IORegistry()

