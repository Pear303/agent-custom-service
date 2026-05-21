import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from api.database import Database

async def test():
    db = Database()
    await db.init()
    
    import aiosqlite
    cursor = await db._pool.execute("SELECT user_id, COUNT(*) as cnt FROM tickets GROUP BY user_id")
    rows = await cursor.fetchall()
    print(f"Total users with tickets: {len(rows)}")
    for user_id, cnt in rows:
        print(f"  User: {user_id}, Tickets: {cnt}")
    
    cursor = await db._pool.execute("SELECT ticket_id, user_id, status, project_name FROM tickets ORDER BY created_at DESC LIMIT 5")
    rows = await cursor.fetchall()
    print(f"\nLatest 5 tickets:")
    for tid, uid, status, name in rows:
        print(f"  - {tid} ({uid}): {status} - {name}")
    
    await db.close()

asyncio.run(test())
