"""psych-cluster: Kura-style psychological taxonomy clustering for GUIDE.

Quick start:
    import asyncio
    from psych_cluster import build_taxonomy, assign_session, load_taxonomy

    # Build once from a corpus of conversations
    taxonomy = asyncio.run(build_taxonomy(
        chunks=[{"id": "s1", "text": "I'm overwhelmed by my dissertation deadline..."}],
        openai_api_key="sk-...",
        objective="the underlying psychological stressor, coping pattern, or support need",
    ))
    taxonomy.save("data/my_taxonomy.json")

    # Load and assign new sessions
    taxonomy = load_taxonomy("data/my_taxonomy.json")
    assignment = asyncio.run(assign_session("session-123", "I feel like a fraud...", taxonomy))
    print(assignment.cluster_name, assignment.confidence)
"""

from .assign import assign_session, assign_sessions
from .cluster import build_taxonomy
from .types import Assignment, Cluster, Taxonomy

def load_taxonomy(path: str) -> Taxonomy:
    return Taxonomy.load(path)

__all__ = [
    "build_taxonomy",
    "assign_session",
    "assign_sessions",
    "load_taxonomy",
    "Taxonomy",
    "Cluster",
    "Assignment",
]
