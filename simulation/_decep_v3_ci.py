"""Redo deception win share the auction-consistent way:
point estimate = MEAN of per-episode win fractions (rounds won / 12), unit = the
EPISODE (n=50), CI = deterministic t interval (df=49). Confirm rounds-won (not
score), and check that every episode really has 12 bookings (so pooled == mean
-of-fractions, as it did in the auction)."""
import json, glob
from collections import defaultdict
import numpy as np

paths = sorted(glob.glob(r"simulation/.runtime/save_slots/save_slot_*/auction_exports/deception_episode/episode_log.json"))
frac = defaultdict(list)          # alias -> per-episode (rounds won / 12)
ep_booking_totals = []
n = 0
for p in paths:
    d = json.load(open(p))
    if not d.get("complete"):
        continue
    n += 1
    aliases = d["selected_models"]
    tot = 0.0
    for ag in d["agents"]:
        frac[ag["alias"]].append(ag["total_reward"] / 12.0)   # rounds won / rounds
        tot += ag["total_reward"]
    ep_booking_totals.append(tot)

T975 = 2.0096  # df=49
print(f"episodes = {n}")
print(f"bookings per episode: min={min(ep_booking_totals):.2f} max={max(ep_booking_totals):.2f} "
      f"mean={np.mean(ep_booking_totals):.2f}  (==12 => pooled equals mean-of-fractions)\n")

ORDER = sorted(frac, key=lambda a: -np.mean(frac[a]))
print(f"{'model':<8}{'win rate':>10}{'t 95% CI':>16}{'pooled%':>10}")
for a in ORDER:
    arr = np.array(frac[a])
    m, sd = arr.mean(), arr.std(ddof=1)
    se = sd / np.sqrt(len(arr))
    lo, hi = m - T975 * se, m + T975 * se
    pooled = arr.sum() / sum(np.array(frac[x]).sum() for x in frac)  # model total / grand total
    print(f"{a:<8}{m*100:>9.1f}%  [{lo*100:4.1f},{hi*100:4.1f}]{pooled*100:>9.1f}")
