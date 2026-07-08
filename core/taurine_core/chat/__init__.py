"""Chat with the knowledge base (spec 2026-07-08-chat-knowledge-base).

retrieval.py — Stage A: latent context pack (heads → RRF → per-chat seed
accumulator → PPR → blend), zero LLM. tools.py — Stage B tool surface
(search_notes / read_note / expand_note). engine.py — the streamed GLM-5.2
thinking turn loop.
"""
