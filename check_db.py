import sqlite3

db = sqlite3.connect(r'C:\Github\ImpulsiveSoundDetection\logs\host.db')

print('=== TABLES ===')
for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    print(' ', row[0])

print()
print('=== detection_events COLUMNS ===')
for row in db.execute('PRAGMA table_info(detection_events)'):
    print(' ', row[1], row[2])

print()
print('=== ROWS ===')
for row in db.execute('SELECT * FROM detection_events'):
    print(' ', row)

print()
print('=== node_status ===')
for row in db.execute('SELECT * FROM node_status'):
    print(' ', row)

print()
print('=== rms_frames count ===')
print(' ', db.execute('SELECT COUNT(*) FROM rms_frames').fetchone()[0])