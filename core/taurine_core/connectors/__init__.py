"""Connector framework for local and future OAuth document sources."""

from .apple_notes import AppleNotesConnector
from .registry import ConnectorRegistry, build_connectors

__all__ = ["AppleNotesConnector", "ConnectorRegistry", "build_connectors"]
