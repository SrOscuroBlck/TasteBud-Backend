from sqlmodel import Session, select
from config.database import engine
from models import MenuItem, Restaurant

with Session(engine) as s:
    rest = s.exec(select(Restaurant).where(Restaurant.name.contains("Crepe"))).first()
    if rest:
        print(f"Restaurant: {rest.name} (id: {rest.id})")
        items = s.exec(select(MenuItem).where(MenuItem.restaurant_id == rest.id)).all()
        courses = {}
        for item in items:
            c = (item.course or "NONE").lower()
            if c not in courses:
                courses[c] = []
            courses[c].append(item.name)
        for c, names in sorted(courses.items()):
            print(f"\n--- {c} ({len(names)} items) ---")
            for n in names[:5]:
                print(f"  {n}")
            if len(names) > 5:
                print(f"  ... and {len(names)-5} more")
        
        print("\n\n=== Items with 'jugo' or 'juice' in name ===")
        for item in items:
            if "jugo" in item.name.lower() or "juice" in item.name.lower():
                print(f"  {item.name} | course: {item.course!r}")
    else:
        print("Restaurant not found")
