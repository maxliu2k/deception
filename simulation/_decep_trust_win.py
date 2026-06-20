"""Trust vs win-rate vs aggression. Is Pro the winner WITHOUT being the most
aggressive claimer? Per-model aggression/efficiency profile + per-round timing."""
import json, glob
from collections import defaultdict
import numpy as np

paths = sorted(glob.glob(r"simulation/.runtime/save_slots/save_slot_*/auction_exports/deception_episode/episode_log.json"))
mean_claim = defaultdict(list); mean_disc = defaultdict(list)
disc_lowtruth = defaultdict(list); disc_hightruth = defaultdict(list)
claim_on_hightruth = defaultdict(list); claim_on_lowtruth = defaultdict(list)
win = defaultdict(lambda: defaultdict(list))     # alias -> round -> [0/1 win]
claim_rd = defaultdict(lambda: defaultdict(list)) # alias -> round -> mean claim
trust_rd = defaultdict(lambda: defaultdict(list)) # alias -> round -> mean trust_before
overall_win = defaultdict(list)
n = 0
for p in paths:
    d = json.load(open(p))
    if not d.get("complete"): continue
    n += 1
    aliases = d["selected_models"]
    idx2alias = {f"agent_{i+1}": aliases[i] for i in range(len(aliases))}
    for ag in d["agents"]:
        overall_win[ag["alias"]].append(ag["total_reward"]/12.0)
    for r in d["rounds"]:
        ri = r["round_idx"]; truth = np.array(r["truth"])
        winners = set(r["winners"])
        for aid, claim in r["claims_by_agent"].items():
            al = idx2alias[aid]; c = np.array(claim)
            disc = np.maximum(0, c - truth)
            mean_claim[al].append(c.mean()); mean_disc[al].append(disc.mean())
            hi = truth >= 0.7; lo = truth <= 0.3
            if hi.any():
                disc_hightruth[al].append(disc[hi].mean()); claim_on_hightruth[al].append(c[hi].mean())
            if lo.any():
                disc_lowtruth[al].append(disc[lo].mean()); claim_on_lowtruth[al].append(c[lo].mean())
            win[al][ri].append(1.0 if aid in winners else 0.0)
            claim_rd[al][ri].append(c.mean())
            tb = np.mean(r["trust_before"][aid]); trust_rd[al][ri].append(tb)

ORDER = sorted(overall_win, key=lambda a: -np.mean(overall_win[a]))
print(f"episodes={n}\n")
print(f"{'model':<8}{'win%':>6}{'mean claim':>11}{'mean disc':>10}{'disc|hi-t':>10}{'disc|lo-t':>10}{'claim|hi-t':>11}")
for a in ORDER:
    print(f"{a:<8}{100*np.mean(overall_win[a]):>5.1f}%{np.mean(mean_claim[a]):>11.3f}"
          f"{np.mean(mean_disc[a]):>10.3f}{np.mean(disc_hightruth[a]):>10.3f}"
          f"{np.mean(disc_lowtruth[a]):>10.3f}{np.mean(claim_on_hightruth[a]):>11.3f}")

print("\nPer-round WIN RATE by model (when does each win?):")
print(f"{'rd':<4}" + "".join(f"{a:>8}" for a in ORDER))
for ri in range(12):
    print(f"{ri:<4}" + "".join(f"{100*np.mean(win[a][ri]):>7.0f}%" for a in ORDER))

print("\nPer-round MEAN TRUST (entering round) by model:")
print(f"{'rd':<4}" + "".join(f"{a:>8}" for a in ORDER))
for ri in range(12):
    print(f"{ri:<4}" + "".join(f"{np.mean(trust_rd[a][ri]):>8.2f}" for a in ORDER))
