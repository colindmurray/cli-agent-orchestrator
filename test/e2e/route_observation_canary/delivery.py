"""Fresh-process delivery driver for one route-observation wake claim.

The route operation writes its wake directly into the real inbox table.  This
driver reopens the same isolated installed store, asks the ordinary
``InboxService`` to deliver that exact row, and records the resulting durable
status.  It creates no second message and has no provider-specific shortcut.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cli_agent_orchestrator.clients import database
from cli_agent_orchestrator.services.inbox_service import inbox_service


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def deliver(receiver_id: str, message_id: int, output_path: Path) -> None:
    database.init_db()
    inbox_service.deliver_pending(
        receiver_id,
        num_messages=1,
        required_message_id=message_id,
    )
    with database.SessionLocal() as session:
        row = (
            session.query(database.InboxModel)
            .filter(database.InboxModel.id == message_id)
            .one_or_none()
        )
        if row is None:
            raise RuntimeError(f"wake inbox row {message_id} disappeared during delivery")
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
    deliver(args.receiver_id, args.message_id, args.output)


if __name__ == "__main__":
    main()
