import json
import subprocess
import sys

# Get a single item to check image_url
result = subprocess.run(
    ["curl", "-s", f"http://localhost:8010/api/v1/restaurants/70f31404-c5d4-4428-a8df-09632603b911"],
    capture_output=True, text=True
)
data = json.loads(result.stdout)
print("Restaurant response keys:", list(data.keys()))
print()

# Check provenance of items in DB
result2 = subprocess.run(
    ["docker", "exec", "tastebud_api", "python", "-c", """
import sys; sys.path.insert(0, '.')
import json
from config.database import get_session
from models.restaurant import MenuItem
from sqlmodel import select

session = next(get_session())
items = session.exec(
    select(MenuItem).where(MenuItem.restaurant_id == '70f31404-c5d4-4428-a8df-09632603b911').limit(10)
).all()

for item in items:
    prov = item.provenance or {}
    img = prov.get('image_url', 'NONE')
    print(f"{item.name}: image_url={img}")

print(f"\\nTotal items checked: {len(items)}")
print(f"Items with images: {sum(1 for i in items if (i.provenance or {}).get('image_url'))}")
"""],
    capture_output=True, text=True
)
# Filter out SQLAlchemy noise
for line in result2.stdout.split('\n'):
    if not line.startswith('2026-') and line.strip():
        print(line)

if result2.returncode != 0:
    print("STDERR:", result2.stderr[:500])
