"""Fresh-process delivery evidence reader for one route-observation wake.

The route operation writes its wake directly into the real inbox table.  This
reader reopens the same isolated installed store after the API server has
delivered the exact row and records its durable status.  It never initiates
delivery: the already-running server is the bridge's pinned controller and is
therefore the only process in this canary allowed to perform provider I/O.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cli_agent_orchestrator.clients import database


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_delivery(receiver_id: str, message_id: int, output_path: Path) -> None:
    with database.SessionLocal() as session:
        row = (
            session.query(database.InboxModel)
            .filter(
                database.InboxModel.id == message_id,
                database.InboxModel.receiver_id == receiver_id,
            )
            .one_or_none()
        )
        if row is None:
            raise RuntimeError(
                f"wake inbox row {message_id} for receiver {receiver_id!r} is absent"
            )
        evidence = {
            "schema": "cao-m17-route-observation-delivery-evidence-v1",
            "message_id": message_id,
            "sender_id": row.sender_id,
            "receiver_id": row.receiver_id,
            "expected_receiver_generation": row.expected_receiver_generation,
            "message_sha256": row.message_sha256,
            "status": row.status,
        }
    _write(output_path, evidence)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-id", required=True)
    parser.add_argument("--message-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read_delivery(args.receiver_id, args.message_id, args.output)


if __name__ == "__main__":
    main()
