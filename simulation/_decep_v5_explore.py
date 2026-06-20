"""Map the v5 deception folders to slots and inspect schema + seat composition."""
import json, glob
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

WANT = ["folder_65", "folder_84", "folder_85", "folder_86", "folder_87",
        "folder_88", "folder_89", "folder_92", "folder_93", "folder_94",
        "folder_95", "folder_96"]


def dec_log_path(sid):
    return HERE / ".runtime" / "save_slots" / sid / "auction_exports" / "deception_episode" / "episode_log.json"


for fid in WANT:
    sids = byfolder.get(fid, [])
    # find first slot with a deception log
    sample = None
    for sid in sids:
        if dec_log_path(sid).exists():
            sample = sid; break
    print(f"{fid:<11} {folders[fid]!r:<40} slots={len(sids)} log_sample={sample}")

# schema peek: one v5 real episode (folder_65) and one transfer (folder_92)
for fid in ["folder_65", "folder_85", "folder_92"]:
    for sid in byfolder.get(fid, []):
        p = dec_log_path(sid)
        if p.exists():
            d = json.load(open(p))
            print(f"\n=== {fid} ({folders[fid]}) slot {sid} ===")
            print("top keys:", list(d.keys()))
            print("selected_models:", d.get("selected_models"))
            print("num_rounds:", d.get("num_rounds"), "complete:", d.get("complete"))
            print("agent0 keys:", list(d["agents"][0].keys()))
            print("agent0:", {k: d["agents"][0][k] for k in d["agents"][0] if k not in ()})
            r0 = d["rounds"][0]
            print("round0 keys:", list(r0.keys()))
            print("round0 truth:", r0.get("truth"))
            tb = r0.get("trust_before", {})
            k0 = next(iter(tb)) if tb else None
            print("trust_before[a0]:", tb.get(k0) if k0 else None, "(scalar or vector?)")
            break
