"""Shared evidence-first retrieval subsystem.

Keep package initialization dependency-free: index construction imports
``retrieval.types`` and must not eagerly load the evidence engine back through
the knowledge store.
"""

__all__: list[str] = []
