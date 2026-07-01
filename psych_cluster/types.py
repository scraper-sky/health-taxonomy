from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Cluster:
    id: str
    name: str
    description: str
    parent_id: str | None
    member_ids: list[str]  # conversation/session IDs in this cluster


@dataclass
class Taxonomy:
    clusters: list[Cluster]
    objective: str = ""

    def leaves(self) -> list[Cluster]:
        parent_ids = {c.parent_id for c in self.clusters if c.parent_id}
        return [c for c in self.clusters if c.id not in parent_ids]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "objective": self.objective,
            "clusters": [
                {
                    "id": c.id,
                    "name": c.name,
                    "description": c.description,
                    "parent_id": c.parent_id,
                    "member_ids": c.member_ids,
                }
                for c in self.clusters
            ],
        }, indent=2))

    @staticmethod
    def load(path: str | Path) -> "Taxonomy":
        raw = json.loads(Path(path).read_text())
        clusters = [
            Cluster(
                id=c["id"],
                name=c["name"],
                description=c["description"],
                parent_id=c.get("parent_id"),
                member_ids=c.get("member_ids", []),
            )
            for c in raw["clusters"]
        ]
        return Taxonomy(clusters=clusters, objective=raw.get("objective", ""))


@dataclass
class Assignment:
    session_id: str
    cluster_id: str | None       # None = no clean fit
    cluster_name: str | None
    confidence: int              # 1-5, 5 = perfect fit
