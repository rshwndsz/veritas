import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from attr import dataclass

from veritas.kb.chunk import Chunk

logger = logging.getLogger("veritas")


@dataclass
class RerankingConfig:
    k: int = 5
    max_tokens: Optional[int] = 8192


async def rerank_chunks(
    query: str,
    chunks: List[Chunk],
    client: httpx.AsyncClient,
    model_id: str,
    cfg: RerankingConfig,
) -> List[Chunk]:
    if not chunks:
        return []

    # https://docs.vllm.ai/en/v0.7.1/serving/openai_compatible_server.html#re-rank-api
    # https://jina.ai/reranker/
    data = {
        "model": model_id,
        "query": query,
        "documents": [chunk.content for chunk in chunks],
        "top_n": cfg.k,
        "return_documents": False,
        "max_tokens": cfg.max_tokens,
    }

    try:
        response = await client.post("rerank", json=data)
        response.raise_for_status()
        response_data = response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"[RRNK][ERROR] HTTP error occurred: {e.response.status_code} - {e.response.text}")
        return []
    except httpx.RequestError as e:
        logger.error(f"[RRNK][ERROR] Request error occurred: {e}")
        return []
    except json.JSONDecodeError:
        logger.error("[RRNK][ERROR] Failed to parse JSON response")
        return []
    except Exception as e:
        logger.error(f"[RRNK][ERROR] An unexpected error occurred: {e}")
        return []

    try:
        results: List[Dict[str, Any]] = response_data.get("results", [])

        # Sort results by relevance_score in descending order
        results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

        # Map sorted indices back to the original chunks
        reranked_chunks = []
        for result in results[: cfg.k]:  # Limit to top_n results
            original_index = result.get("index")
            if original_index is not None and 0 <= original_index < len(chunks):
                # Ensure the index is valid for the original chunks list
                reranked_chunks.append(chunks[original_index])
            else:
                logger.warning(f"[RRNK][WARN] Invalid index {original_index} received from reranker.")

        if len(reranked_chunks) < cfg.k and len(results) > 0:
            logger.warning(
                f"[RRNK][WARN] Reranker returned fewer valid results ({len(reranked_chunks)}) than requested ({cfg.k})."
            )

        return reranked_chunks

    except KeyError as e:
        logger.error(f"[RRNK][ERROR] Missing key in response data: {e} - Data: {response_data}")
        return []
    except Exception as e:
        logger.error(f"[RRNK][ERROR] Error processing reranker results: {e}")
        return []
