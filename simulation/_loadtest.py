import sys, json, pickle
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "simulation")

HERE = Path("simulation")
sid = "save_slot_840"
try:
    obj = pickle.load(open(HERE / ".runtime" / "save_slots" / sid / "runtime.pkl", "rb"))
    print("LOADED OK; top keys:", list(obj.keys()))
    reset = obj.get("last_reset", {})
    print("selected_models:", reset.get("selected_models"))
    der = obj.get("last_result", {}).get("derived", {})
    cp = der.get("completed_paintings", [])
    print("num completed_paintings:", len(cp))
    if cp:
        print("sample painting keys:", list(cp[0].keys()))
        print("sample winner_id:", cp[0].get("winner_id"))
except Exception as e:
    import traceback
    traceback.print_exc()
