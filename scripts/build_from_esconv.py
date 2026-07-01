"""One-time script: build a psychological taxonomy from ESConv and save it.

Usage:
    OPENAI_API_KEY=sk-... python scripts/build_from_esconv.py
    OPENAI_API_KEY=sk-... python scripts/build_from_esconv.py --n 300 --out data/esconv_small.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from psych_cluster import build_taxonomy

OBJECTIVE = (
    "the underlying psychological stressor, coping pattern, or support need "
    "the person is experiencing"
)


def load_esconv(path: Path, n: int | None) -> list[dict]:
    raw = json.loads(path.read_text())
    chunks = []
    for rec in raw:
        msgs = rec.get("chat_messages") or []
        user_text = " ".join(
            m["text"] for m in msgs if m.get("sender") == "human" and m.get("text")
        )
        if not user_text.strip():
            continue
        chunks.append({"id": rec["uuid"], "text": user_text})
        if n and len(chunks) >= n:
            break
    return chunks


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esconv", type=Path,
                    default=Path(__file__).parent.parent.parent /
                    "taxonomy-testing/data_runtime/datasets/esconv.json",
                    help="path to esconv.json (default: ../taxonomy-testing/data_runtime/datasets/esconv.json)")
    ap.add_argument("--n", type=int, default=None, help="max conversations to use (default: all)")
    ap.add_argument("--out", type=Path, default=Path("data/esconv_taxonomy.json"))
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("error: set OPENAI_API_KEY environment variable")
        sys.exit(1)

    if not args.esconv.exists():
        print(f"error: ESConv file not found at {args.esconv}")
        print("run taxonomy-testing/scripts/import_esconv.py first")
        sys.exit(1)

    chunks = load_esconv(args.esconv, args.n)
    print(f"loaded {len(chunks)} conversations from ESConv")

    async def log(msg: str) -> None:
        print(msg)

    taxonomy = await build_taxonomy(
        chunks,
        openai_api_key=api_key,
        objective=OBJECTIVE,
        log=log,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    taxonomy.save(args.out)
    print(f"\nwrote taxonomy → {args.out}")
    print(f"  {len(taxonomy.clusters)} clusters ({len(taxonomy.leaves())} leaves)")
    print("\nLeaf clusters:")
    for leaf in taxonomy.leaves():
        print(f"  [{len(leaf.member_ids):3d}] {leaf.name}")


if __name__ == "__main__":
    asyncio.run(main())
