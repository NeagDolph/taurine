from __future__ import annotations

from .claude_memories import ClaudeMemoriesConnector
from .codex_memories import CodexMemoriesConnector
from .gmail import GmailConnector
from .imessage import IMessageConnector
from .obsidian import ObsidianConnector
from .outlook import OutlookConnector


CONNECTORS: list = [
    ClaudeMemoriesConnector(),
    CodexMemoriesConnector(),
    ObsidianConnector(),
    IMessageConnector(),
    GmailConnector(),
    OutlookConnector(),
]

try:
    from .apple_notes import AppleNotesConnector

    CONNECTORS.insert(0, AppleNotesConnector())
except ImportError:
    pass
