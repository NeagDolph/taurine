"""Stage B — the streamed GLM-5.2 thinking turn loop (spec chat §1.3–1.4).

One ChatEngine per server; turns are per-chat single-flight with
CHAT_MAX_CONCURRENT_TURNS across chats. A turn:

1. persist the user message; create the empty assistant row (→ chat_ack)
2. Stage-A context pack (→ chat_retrieval)
3. stream_chat(CHAT_MODEL, reasoning=CHAT_REASONING, tools=TOOL_DEFINITIONS):
   reasoning deltas → chat_thinking, content deltas → chat_delta; on
   tool_call: emit chat_activity(running), execute, emit chat_activity(done),
   append tool message, re-open the stream — until no tool calls or
   CHAT_MAX_TOOL_CALLS
4. validate citations (lenient: strip unknown [N#]/[F#], never reject),
   resolve sources, finish_chat_message, save retriever state (→ chat_done)
5. auto-title after the first completed exchange (one flash complete_json;
   fallback: first 40 chars) (→ chat_meta)

Cancellation (chat_cancel): cancel the turn task; partial content persists
with cancelled=True (→ chat_done with cancelled). Fallback model only on
stream-OPEN failure; mid-stream errors → chat_error. Session-log record
kinds: chat_turn, chat_retrieval.
"""
from __future__ import annotations

from typing import Any, Callable, Coroutine

from ..indexer.store import Index
from ..llm import OpenRouterClient

# async callback the server provides: emits one ws event dict/model to the client
EmitFn = Callable[[Any], Coroutine[Any, Any, None]]


class ChatEngine:
    def __init__(self, index: Index, llm: OpenRouterClient, logger: Any) -> None:
        self.index = index
        self.llm = llm
        self.logger = logger
        raise NotImplementedError  # Stage B

    async def send(self, chat_id: str, text: str, emit: EmitFn) -> None:
        """Handle chat_send ("" chat_id → create). Runs the full turn;
        busy/limit violations emit chat_error(code="busy") and return."""
        raise NotImplementedError  # Stage B

    async def cancel(self, chat_id: str) -> None:
        raise NotImplementedError  # Stage B

    async def regenerate(self, chat_id: str, emit: EmitFn) -> None:
        """Delete the last assistant message and re-run the last user one."""
        raise NotImplementedError  # Stage B
