# show_db.py
import sqlite3
conn = sqlite3.connect("attendance.db")
c = conn.cursor()
print("Students:")
for r in c.execute("SELECT id, roll, name FROM students"):
    print(r)
print("\nAttendance (last 50):")
for r in c.execute("SELECT id, student_id, ts, status FROM attendance ORDER BY ts DESC LIMIT 50"):
    print(r)
conn.close()
