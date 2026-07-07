from __future__ import annotations

from .claude_memories import ClaudeMemoriesConnector
from .codex_memories import CodexMemoriesConnector
from .imessage import IMessageConnector
from .obsidian import ObsidianConnector


CONNECTORS: list = [
    ClaudeMemoriesConnector(),
    CodexMemoriesConnector(),
    ObsidianConnector(),
    IMessageConnector(),
]

try:
    from .apple_notes import AppleNotesConnector

    CONNECTORS.insert(0, AppleNotesConnector())
except ImportError:
    pass
