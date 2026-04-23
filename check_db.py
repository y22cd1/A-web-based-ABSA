import sqlite3
import os
import json

db_path = os.path.join(r"c:\Users\Sunny\Desktop\FOLDERS\VS FILES\ISPROJECT\customer-review-insight-AI", 'instance', 'site.db')
print('Checking DB at:', db_path)
if not os.path.exists(db_path):
    print('DB file not found')
else:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name, type, sql FROM sqlite_master WHERE type IN ('table','index') ORDER BY name;")
    rows = cur.fetchall()
    if not rows:
        print('No tables found in DB')
    else:
        print('Found tables/indexes:')
        for name, typ, sql in rows:
            print('-', name, typ)
    conn.close()
