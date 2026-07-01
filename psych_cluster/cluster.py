"""Kura-style clustering: embed → k-means → LLM name → meta-cluster.

Stripped from taxonomy-testing/app/clustering/kura.py with all web-app
dependencies removed. Takes plain text chunks, returns a Taxonomy.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Awaitable, Callable

import instructor
import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

from .types import Cluster, Taxonomy

LogFn = Callable[[str], Awaitable[None]]


async def _noop_log(_: str) -> None:
    pass


# ----------------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------------

async def _embed(texts: list[str], api_key: str, model: str, log: LogFn) -> np.ndarray:
    client = AsyncOpenAI(api_key=api_key)
    BATCH = 96
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = [t or " " for t in texts[i: i + BATCH]]
        resp = await client.embeddings.create(model=model, input=batch)
        out.extend(d.embedding for d in resp.data)
        await log(f"  embeddings: {min(i + BATCH, len(texts))}/{len(texts)}")
    return np.asarray(out, dtype=np.float32)


# ----------------------------------------------------------------------------
# k-means
# ----------------------------------------------------------------------------

TARGET_PER_CLUSTER = 10
MIN_K = 3
MAX_K = 25


def _pick_k(n: int) -> int:
    return max(MIN_K, min(MAX_K, round(n / TARGET_PER_CLUSTER)))


def _base_clusters(embeddings: np.ndarray) -> tuple[np.ndarray, int]:
    n = embeddings.shape[0]
    k = min(_pick_k(n), n)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalised = embeddings / norms
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    return km.fit_predict(normalised), k


# ----------------------------------------------------------------------------
# LLM naming
# ----------------------------------------------------------------------------

class _ClusterName(BaseModel):
    name: str = Field(description="short imperative-style label, ≤12 words")
    description: str = Field(description="one sentence describing what these conversations have in common")


NAME_PROMPT = """Here are summaries of {n} related conversations grouped by embedding similarity.

CONVERSATIONS:
{joined}

Give the group a short imperative-style NAME (≤12 words) and a one-sentence DESCRIPTION of what they share.
Be specific — avoid generic words like "various", "general", "diverse".{objective_suffix}"""


class _MetaLabel(BaseModel):
    name: str
    description: str
    children: list[str] = Field(description="exact leaf names that roll up under this top-level")


class _MetaTaxonomy(BaseModel):
    top_labels: list[_MetaLabel]


META_PROMPT = """Here is a flat list of {n} cluster labels with member counts:

{joined}

Group these into 3–7 broader TOP-LEVEL themes. Every leaf must roll up under exactly one top-level.
Top-level names should be imperative phrases that capture what the leaves under them share."""


async def _name_one(client, sem, members: list[str], objective_suffix: str) -> _ClusterName:
    sample = members[:10]
    joined = "\n".join(f"- {s[:240]}" for s in sample)
    async with sem:
        return await client.chat.completions.create(
            messages=[{"role": "user", "content": NAME_PROMPT.format(
                n=len(members), joined=joined, objective_suffix=objective_suffix,
            )}],
            response_model=_ClusterName,
        )


async def _name_clusters(
    group_texts: dict[int, list[str]],
    clustering_llm: str,
    objective: str | None,
    log: LogFn,
) -> dict[int, _ClusterName]:
    client = instructor.from_provider(clustering_llm, async_client=True)
    sem = asyncio.Semaphore(8)
    suffix = (f" Where the group reflects {objective.strip()}, foreground that in the name."
              if objective and objective.strip() else "")
    keys = sorted(group_texts)
    results = await asyncio.gather(*[_name_one(client, sem, group_texts[k], suffix) for k in keys])
    await log(f"  named {len(results)} base clusters")
    return dict(zip(keys, results))


async def _meta_cluster(named: dict[int, _ClusterName], counts: dict[int, int], clustering_llm: str) -> _MetaTaxonomy:
    if len(named) <= 1:
        return _MetaTaxonomy(top_labels=[])
    client = instructor.from_provider(clustering_llm, async_client=True)
    joined = "\n".join(f"- {n.name} ({counts.get(k, 0)}): {n.description}" for k, n in named.items())
    return await client.chat.completions.create(
        messages=[{"role": "user", "content": META_PROMPT.format(n=len(named), joined=joined)}],
        response_model=_MetaTaxonomy,
    )


# ----------------------------------------------------------------------------
# Public entrypoint
# ----------------------------------------------------------------------------

def _hid(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


async def build_taxonomy(
    chunks: list[dict],
    *,
    openai_api_key: str,
    clustering_llm: str = "openai/gpt-4o-mini",
    embed_model: str = "text-embedding-3-small",
    objective: str | None = None,
    log: LogFn | None = None,
) -> Taxonomy:
    """Build a Taxonomy from a list of {"id": str, "text": str} chunks.

    Args:
        chunks: list of {"id": ..., "text": ...} dicts
        openai_api_key: OpenAI API key for embeddings
        clustering_llm: instructor provider string for LLM naming (default gpt-4o-mini)
        embed_model: OpenAI embedding model
        objective: optional lens to steer cluster naming (not grouping)
        log: optional async logging callback
    """
    _log = log or _noop_log
    if not chunks:
        return Taxonomy(clusters=[], objective=objective or "")

    await _log(f"psych-cluster: embedding {len(chunks)} chunks with {embed_model}")
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    embeddings = await _embed(texts, openai_api_key, embed_model, _log)

    labels, k_used = _base_clusters(embeddings)
    await _log(f"psych-cluster: k-means k={k_used}")

    groups: dict[int, list[str]] = {}
    group_texts: dict[int, list[str]] = {}
    for cid, text, lab in zip(ids, texts, labels):
        groups.setdefault(int(lab), []).append(cid)
        group_texts.setdefault(int(lab), []).append(text)

    named = await _name_clusters(group_texts, clustering_llm, objective, _log)
    counts = {k: len(v) for k, v in groups.items()}

    await _log("psych-cluster: meta-clustering leaf labels")
    meta = await _meta_cluster(named, counts, clustering_llm)

    leaf_id = {k: _hid(f"leaf-{k}-{named[k].name}") for k in named}
    out: list[Cluster] = []
    used: set[int] = set()

    if meta.top_labels:
        name_to_key = {named[k].name: k for k in named}
        for top in meta.top_labels:
            kids = [name_to_key[c] for c in top.children if c in name_to_key]
            member_ids: list[str] = []
            for k in kids:
                member_ids.extend(groups[k])
                used.add(k)
            top_id = _hid(f"top-{top.name}")
            out.append(Cluster(id=top_id, name=top.name, description=top.description,
                               parent_id=None, member_ids=member_ids))
            for k in kids:
                n = named[k]
                out.append(Cluster(id=leaf_id[k], name=n.name, description=n.description,
                                   parent_id=top_id, member_ids=groups[k]))

    orphans = [k for k in named if k not in used]
    if orphans:
        rid = _hid("residual")
        rmembers: list[str] = []
        for k in orphans:
            rmembers.extend(groups[k])
        out.append(Cluster(id=rid, name="Other / unassigned",
                           description="Clusters not placed by meta-pass.",
                           parent_id=None, member_ids=rmembers))
        for k in orphans:
            n = named[k]
            out.append(Cluster(id=leaf_id[k], name=n.name, description=n.description,
                               parent_id=rid, member_ids=groups[k]))

    return Taxonomy(clusters=out, objective=objective or "")
