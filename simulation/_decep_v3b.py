"""Follow-up: per-model win rate in the FINAL round vs overall, and mean trust
entering the final round. Tests whether Llama's low-claim conservatism at least
banks trust (the 'hoard for the endgame' payoff)."""
import json, glob
from collections import defaultdict
import numpy as np

paths = sorted(glob.glob(r"simulation/.runtime/save_slots/save_slot_*/auction_exports/deception_episode/episode_log.json"))
final_reward = defaultdict(list); overall_reward = defaultdict(list)
trust_into_final = defaultdict(list)
n = 0
for p in paths:
    d = json.load(open(p))
    if not d.get("complete"): continue
    n += 1
    aliases = d["selected_models"]
    idx2alias = {f"agent_{i+1}": aliases[i] for i in range(len(aliases))}
    T = len(d["rounds"])
    last = d["rounds"][T-1]
    for aid, al in idx2alias.items():
        final_reward[al].append(last["rewards_by_agent"].get(aid, 0.0))
        tb = last["trust_before"][aid]
        trust_into_final[al].append(float(np.mean(tb)))
    for ag in d["agents"]:
        overall_reward[ag["alias"]].append(ag["total_reward"] / 12.0)

ORDER = sorted(overall_reward, key=lambda a: -np.mean(overall_reward[a]))
print(f"episodes={n}\n")
print(f"{'model':<8}{'overall win%':>13}{'FINAL-round win%':>18}{'mean trust into final':>24}")
for a in ORDER:
    ov = 100*np.mean(overall_reward[a])
    fr = 100*np.mean(final_reward[a])
    tr = np.mean(trust_into_final[a])
    print(f"{a:<8}{ov:>12.1f}%{fr:>17.1f}%{tr:>24.3f}")
