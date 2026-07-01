"""Assign a new session to the nearest leaf in a frozen Taxonomy."""

from __future__ import annotations

import asyncio

import instructor
from pydantic import BaseModel, Field
from tenacity import AsyncRetrying, stop_after_attempt, wait_random_exponential

from .types import Assignment, Taxonomy

NONE_LABEL = "NONE"

ASSIGN_PROMPT = """You are assigning a new conversation to a fixed taxonomy of psychological support clusters.

TAXONOMY (name: description):
{taxonomy}

Rules:
- Pick the single best-fitting label_name (exact match to a name above).
- If multiple fit, pick the most specific one.
- If nothing fits cleanly, answer {none}.

CONVERSATION:
{text}"""


class _LeafChoice(BaseModel):
    label_name: str = Field(
        description=f"exactly one label name from the taxonomy, or {NONE_LABEL} if nothing fits"
    )
    confidence: int = Field(ge=1, le=5, description="5 = perfect fit, 1 = forced fit")


def _retrying() -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(6),
        wait=wait_random_exponential(multiplier=1, max=60),
        reraise=True,
    )


async def assign_session(
    session_id: str,
    session_text: str,
    taxonomy: Taxonomy,
    *,
    clustering_llm: str = "openai/gpt-4o-mini",
) -> Assignment:
    """Assign a single session to the best-fitting leaf in the taxonomy.

    Args:
        session_id: unique ID for this session (e.g. GUIDE session UUID)
        session_text: the full text of the session (user messages, transcript, or summary)
        taxonomy: a frozen Taxonomy built by build_taxonomy() or loaded from disk
        clustering_llm: instructor provider string

    Returns:
        Assignment with cluster_id, cluster_name, and confidence (1-5)
    """
    leaves = taxonomy.leaves()
    if not leaves:
        return Assignment(session_id=session_id, cluster_id=None, cluster_name=None, confidence=1)

    tax_str = "\n".join(f"- {l.name}: {l.description}" for l in leaves)
    prompt = ASSIGN_PROMPT.format(taxonomy=tax_str, none=NONE_LABEL, text=session_text[:2000])
    by_name = {l.name.strip().lower(): l for l in leaves}

    client = instructor.from_provider(clustering_llm, async_client=True)
    sem = asyncio.Semaphore(1)

    async with sem:
        choice: _LeafChoice = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            response_model=_LeafChoice,
            max_retries=_retrying(),
        )

    name = choice.label_name.strip()
    if name.upper() == NONE_LABEL:
        return Assignment(session_id=session_id, cluster_id=None, cluster_name=None,
                          confidence=choice.confidence)

    leaf = by_name.get(name.lower())
    if leaf is None:
        # fuzzy fallback
        target = name.lower()
        leaf = next((l for l in leaves if target in l.name.lower() or l.name.lower() in target), None)

    if leaf is None:
        return Assignment(session_id=session_id, cluster_id=None, cluster_name=None,
                          confidence=1)

    return Assignment(session_id=session_id, cluster_id=leaf.id, cluster_name=leaf.name,
                      confidence=choice.confidence)


async def assign_sessions(
    sessions: list[dict],
    taxonomy: Taxonomy,
    *,
    clustering_llm: str = "openai/gpt-4o-mini",
    parallel: int = 10,
) -> list[Assignment]:
    """Assign multiple sessions in parallel.

    Args:
        sessions: list of {"id": str, "text": str}
        taxonomy: frozen Taxonomy
        clustering_llm: instructor provider string
        parallel: max concurrent LLM calls
    """
    sem = asyncio.Semaphore(parallel)
    leaves = taxonomy.leaves()
    tax_str = "\n".join(f"- {l.name}: {l.description}" for l in leaves)
    by_name = {l.name.strip().lower(): l for l in leaves}
    client = instructor.from_provider(clustering_llm, async_client=True)

    async def _one(session: dict) -> Assignment:
        prompt = ASSIGN_PROMPT.format(
            taxonomy=tax_str, none=NONE_LABEL, text=(session["text"] or "")[:2000]
        )
        async with sem:
            choice: _LeafChoice = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                response_model=_LeafChoice,
                max_retries=_retrying(),
            )
        name = choice.label_name.strip()
        if name.upper() == NONE_LABEL:
            return Assignment(session_id=session["id"], cluster_id=None,
                              cluster_name=None, confidence=choice.confidence)
        leaf = by_name.get(name.lower())
        if leaf is None:
            target = name.lower()
            leaf = next((l for l in leaves if target in l.name.lower() or l.name.lower() in target), None)
        if leaf is None:
            return Assignment(session_id=session["id"], cluster_id=None, cluster_name=None, confidence=1)
        return Assignment(session_id=session["id"], cluster_id=leaf.id,
                          cluster_name=leaf.name, confidence=choice.confidence)

    return await asyncio.gather(*[_one(s) for s in sessions])
