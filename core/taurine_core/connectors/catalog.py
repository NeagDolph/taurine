from __future__ import annotations

from .claude_memories import ClaudeMemoriesConnector
from .codex_memories import CodexMemoriesConnector
from .gmail import GmailConnector
from .google_calendar import GoogleCalendarConnector
from .google_docs import GoogleDocsConnector
from .imessage import IMessageConnector
from .local_folders import LocalFoldersConnector
from .obsidian import ObsidianConnector
from .outlook import OutlookConnector


CONNECTORS: list = [
    ClaudeMemoriesConnector(),
    CodexMemoriesConnector(),
    LocalFoldersConnector(),
    ObsidianConnector(),
    IMessageConnector(),
    GmailConnector(),
    GoogleCalendarConnector(),
    GoogleDocsConnector(),
    OutlookConnector(),
]

try:
    from .apple_notes import AppleNotesConnector

    CONNECTORS.insert(0, AppleNotesConnector())
except ImportError:
    pass
