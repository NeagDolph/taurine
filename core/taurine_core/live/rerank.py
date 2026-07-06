from __future__ import annotations

from typing import Any

import httpx

from taurine_core import config
from taurine_core.ports import RerankError


class CohereReranker:
    def __init__(
        self,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._transport = transport

    async def rerank(
        self,
        query: str,
        docs: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not docs:
            return []

        payload = {
            "model": config.RERANK_MODEL,
            "query": query,
            "documents": [doc_text for _note_id, doc_text in docs],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(
                timeout=config.RERANK_TIMEOUT_S,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    "https://api.cohere.com/v2/rerank",
                    json=payload,
                    headers=headers,
                )
            if response.status_code >= 400:
                raise RerankError(f"Cohere rerank HTTP {response.status_code}")
            ranked = self._parse_response(response.json(), docs)
        except RerankError:
            raise
        except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError) as exc:
            raise RerankError("Cohere rerank failed") from exc

        return sorted(ranked, key=lambda row: row[1], reverse=True)

    def _parse_response(
        self,
        payload: dict[str, Any],
        docs: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        raw_results = payload["results"]
        if not isinstance(raw_results, list):
            raise TypeError("results must be a list")

        ranked: list[tuple[str, float]] = []
        for result in raw_results:
            if not isinstance(result, dict):
                raise TypeError("result must be an object")
            index = result["index"]
            score = result["relevance_score"]
            if not isinstance(index, int):
                raise TypeError("index must be an int")
            note_id = docs[index][0]
            ranked.append((note_id, float(score)))
        return ranked
