"""JSON → SQLite 数据迁移脚本"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import Database


async def migrate():
    db = Database()
    await db.init()

    context_path = Path(__file__).parent.parent / "data" / "user_context.json"
    if not context_path.exists():
        print("无 JSON 数据需要迁移")
        await db.close()
        return

    with open(context_path, "r", encoding="utf-8-sig") as f:
        ctx = json.load(f)

    migrated = 0
    skipped = 0
    for user_id, user_data in ctx.items():
        for ticket in user_data.get("tickets", []):
            ticket_id = ticket.get("ticket_id")
            existing = await db.get_ticket(ticket_id)
            if existing:
                skipped += 1
                continue

            ticket_data = {
                "ticket_id": ticket_id,
                "user_id": user_id,
                "project_name": ticket.get("project_name", ""),
                "project_type": ticket.get("project_type", ""),
                "description": ticket.get("description", ""),
                "deadline": ticket.get("deadline", ""),
                "budget": ticket.get("budget", ""),
            }
            tid = await db.create_ticket(ticket_data)

            update_fields = {}
            if ticket.get("analysis"):
                update_fields["analysis"] = ticket["analysis"]
            if ticket.get("prd"):
                update_fields["prd"] = ticket["prd"]
            if ticket.get("quote"):
                update_fields["quote"] = ticket["quote"]
            if ticket.get("error"):
                update_fields["error"] = ticket["error"]

            status = ticket.get("status", "queued")
            status_map = {
                "design_failed": "failed",
                "analyzed": "completed",
                "estimating": "estimating",
                "designing": "designing",
                "analyzing": "analyzing",
                "completed": "completed",
                "failed": "failed",
            }
            status = status_map.get(status, "queued")
            await db.update_ticket_status(tid, status, **update_fields)
            migrated += 1

    print(f"迁移完成：{migrated} 个新工单，{skipped} 个已存在跳过")
    await db.close()


if __name__ == "__main__":
    asyncio.run(migrate())
