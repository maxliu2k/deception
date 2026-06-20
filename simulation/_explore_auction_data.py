"""Explore the saved-run catalog: list folders, slot counts, and peek at the
seat composition of one slot per folder so we can tell vs-mimic from vs-real."""
import json, pickle
from collections import defaultdict
from pathlib import Path

HERE = Path("simulation")
cat = json.load(open(HERE / ".runtime" / "save_slots.json", encoding="utf-8"))
folders = {f["folder_id"]: f.get("name", "") for f in cat.get("folders", [])}
slots = cat.get("slots", {})
if isinstance(slots, list):
    slots = {s["slot_id"]: s for s in slots}
byfolder = defaultdict(list)
for sid, m in slots.items():
    byfolder[m.get("folder_id")].append(sid)


def load(sid):
    return pickle.load(open(HERE / ".runtime" / "save_slots" / sid / "runtime.pkl", "rb"))


print(f"{'fid':>6}  {'#slots':>6}  name")
print("-" * 60)
for fid in sorted(folders, key=lambda x: (str(type(x)), str(x))):
    sids = byfolder.get(fid, [])
    print(f"{str(fid):>6}  {len(sids):>6}  {folders[fid]!r}")

print("\n\n=== seat composition peek (first loadable slot per folder) ===")
for fid in sorted(folders, key=lambda x: (str(type(x)), str(x))):
    name = folders[fid]
    sids = byfolder.get(fid, [])
    if not sids:
        continue
    peeked = False
    for sid in sids:
        try:
            obj = load(sid)
        except Exception as e:
            continue
        reset = obj.get("last_reset", {})
        models = reset.get("selected_models")
        der = obj.get("last_result", {}).get("derived", {})
        ncp = len(der.get("completed_paintings", []))
        print(f"\nfid={fid!r}  name={name!r}  slot={sid}")
        print(f"   selected_models = {models}")
        print(f"   completed_paintings = {ncp}")
        peeked = True
        break
    if not peeked:
        print(f"\nfid={fid!r}  name={name!r}  -- no loadable slot")
