"""Figure out how slot ids map to runtime.pkl directories."""
import json
from pathlib import Path

HERE = Path("simulation")
cat = json.load(open(HERE / ".runtime" / "save_slots.json", encoding="utf-8"))

print("=== top-level keys ===", list(cat.keys()))
print("\n=== one folder entry ===")
print(json.dumps(cat.get("folders", [])[0], indent=2)[:800])

slots = cat.get("slots", {})
print("\n=== slots type ===", type(slots).__name__)
if isinstance(slots, dict):
    k0 = next(iter(slots))
    print("first slot key:", repr(k0))
    print("first slot value:")
    print(json.dumps(slots[k0], indent=2)[:1000])
else:
    print("first slot entry:")
    print(json.dumps(slots[0], indent=2)[:1000])

sdir = HERE / ".runtime" / "save_slots"
dirs = [p.name for p in sdir.iterdir() if p.is_dir()]
print(f"\n=== {len(dirs)} dirs in save_slots/ (first 10) ===")
for d in dirs[:10]:
    has = (sdir / d / "runtime.pkl").exists()
    print(f"   {d}   runtime.pkl={has}")
