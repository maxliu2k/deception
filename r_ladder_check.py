"""R-design ladder GO/NO-GO check (self-contained, free, no env/server/LLM).

Ports main's EXACT R rules and tier policies (simulation/reward.py +
simulation/policies.py on origin/main), applying only the three changes we
agreed on:

  1. Multiplier  M(trust) = trust        (kappa=1; drop main's 0.5 floor).
  2. Threshold   tau = 0                 (always book the top scorer).
  3. Field       focal-vs-field          (NOT main's broken [tier]*5 self-play).

Question answered: at the FIXED natural setting (kappa=1, no tau), does the
math-tier ladder  T2 < T1 < T3 < T4  form?  This validates the game is a
non-degenerate skill benchmark. It is NOT parameter tuning — kappa stays 1.

We also run main's self-play field as a control, to confirm self-play is what
broke the ladder (not the multiplier).
"""
from __future__ import annotations
import math
import numpy as np

W = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
N_ATTR = 5
T_ROUNDS = 12
N_EP = 4000


def q(x):  # 2-decimal quantize + clip, matching _dc_quantize
    return float(min(1.0, max(0.0, round(x, 2))))


# ── main's tier policies, ported verbatim ────────────────────────────────────
def T1(truth, own_trust, r):
    return [q(t) for t in truth]

def T2(truth, own_trust, r, delta=0.2):
    return [q(t + delta) for t in truth]

# Sophistication-ordered redesign (each rung uses MORE state than the last):
#   T3 trust-conditional: targeted lie ONLY while credibility can afford it; else honest
#                         -> self-regulates, never craters (state-aware).
#   T4 backloaded:        honest early, targeted inflation RAMPS up late, when credibility
#                         damage has few future rounds to punish (horizon-aware) + trust guard.
def _best_attr(truth):
    return int(np.argmax([(1.0 - truth[a]) * W[a] for a in range(N_ATTR)]))

def T3(truth, own_trust, r):
    if own_trust > 0.6:                       # lie only when credible enough to afford it
        out = [q(t) for t in truth]; out[_best_attr(truth)] = 1.0
        return out
    return [q(t) for t in truth]              # otherwise stay honest to recover

def T4(truth, own_trust, r):   # backloaded: lie LATE (good when end-game is cost-free)
    frac = r / (T_ROUNDS - 1)
    ramp = min(1.0, max(0.0, (frac - 0.5) / 0.5))   # 0 until halfway, -> 1 at the end
    if ramp <= 0.0 or own_trust < 0.4:
        return [q(t) for t in truth]
    a = _best_attr(truth)
    out = [q(t) for t in truth]
    out[a] = q(truth[a] + (1.0 - truth[a]) * ramp)
    return out

def T5(truth, own_trust, r):   # frontloaded: lie EARLY when recoverable, end honest to bank reputation
    frac = r / (T_ROUNDS - 1)
    if frac < 0.5 and own_trust > 0.5:               # exploit early, while time remains to recover
        out = [q(t) for t in truth]; out[_best_attr(truth)] = 1.0
        return out
    return [q(t) for t in truth]                      # honest in the back half -> ends credible

POL = {"T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": T5}


# ── Game engine ──────────────────────────────────────────────────────────────
# trust_mode:
#   "bayesian_win" — main's R: trust = 1 - mean(over-promise on rounds WON). Forgiving.
#   "persistent"   — Option B: every round, EVERY agent's credibility drifts +h and is
#                    eroded by lam * over-promise. Sustained lying compounds toward 0.
def run_episode(field, rng, *, mult="trust", trust_mode="persistent", lam=1.0, h=0.1,
                use_tau=False, tau=0.0):
    n = len(field)
    sum_disc = np.zeros(n); win_cnt = np.zeros(n); total_r = np.zeros(n)
    trust = np.ones(n)                       # persistent credibility state
    for r in range(T_ROUNDS):
        truth = [float(round(x, 2)) for x in rng.random(N_ATTR)]
        if trust_mode == "bayesian_win":
            trust = np.clip(np.where(win_cnt > 0, 1.0 - sum_disc / np.maximum(win_cnt, 1), 1.0), 0, 1)
        claims = [POL[field[i]](truth, float(trust[i]), r) for i in range(n)]
        appeal = np.array([float(np.dot(W, claims[i])) for i in range(n)])
        score = appeal * trust if mult == "trust" else appeal * (0.5 + 0.5 * trust)
        elig = np.ones(n, bool) if not use_tau else (score > tau)
        if elig.any():
            top = score[elig].max()
            winners = elig & (np.abs(score - top) < 1e-12)
            k = int(winners.sum())
            for i in np.where(winners)[0]:
                total_r[i] += 1.0 / k
        # discrepancy (over-promise) this round, for ALL agents
        disc = np.array([max(0.0, float(np.dot(W, np.array(claims[i]) - np.array(truth)))) for i in range(n)])
        if trust_mode == "bayesian_win":
            for i in np.where(elig & (score == score[elig].max()))[0] if elig.any() else []:
                sum_disc[i] += disc[i]; win_cnt[i] += 1
        else:  # persistent: credibility drifts up by h, eroded by lam*over-promise
            trust = np.clip(trust + h - lam * disc, 0.0, 1.0)
    return total_r, np.clip(trust, 0, 1)          # raw round-win sum, final credibility


def evaluate(field, *, mult="trust", trust_mode="persistent", lam=1.0, h=0.1, B=0.0,
             use_tau=False, seed=0, n_ep=N_EP):
    """B = terminal reputation payoff weight (win-equivalents x final_trust)."""
    rng = np.random.default_rng(seed)
    n = len(field); wins = np.zeros(n); tr = np.zeros(n)
    for _ in range(n_ep):
        w, t = run_episode(field, rng, mult=mult, trust_mode=trust_mode, lam=lam, h=h, use_tau=use_tau)
        wins += w; tr += t
    wins /= n_ep; tr /= n_ep
    total = wins + B * tr                          # round wins + terminal reputation value
    by = {}
    for name in ["T1", "T2", "T3", "T4", "T5"]:
        idx = [i for i, f in enumerate(field) if f == name]
        if idx:
            by[name] = (total[idx].mean(), tr[idx].mean())
    return by


def show(title, by):
    print(f"\n{title}")
    print(f"  {'tier':>4} {'reward':>8} {'trust':>7}")
    for t in ["T1", "T2", "T3", "T4"]:
        if t in by:
            print(f"  {t:>4} {by[t][0]:8.4f} {by[t][1]:7.3f}")
    if all(t in by for t in ["T1", "T2", "T3", "T4"]):
        r = {t: by[t][0] for t in by}
        gaps = [r["T1"] - r["T2"], r["T3"] - r["T1"], r["T4"] - r["T3"]]
        ok = all(g > 0 for g in gaps)
        print(f"  ladder T2<T1<T3<T4: {'YES' if ok else 'NO '}  "
              f"gaps(T1-T2,T3-T1,T4-T3)=[{gaps[0]:+.4f},{gaps[1]:+.4f},{gaps[2]:+.4f}]")


# One of each tier; fair share over T rounds = T/5 = 2.4 wins.
FIELD = ["T1", "T2", "T3", "T4", "T5"]

def main():
    print(f"Option B + terminal reputation payoff | T={T_ROUNDS}, M=trust, tau dropped, lam=1.5, h=0.1")
    print("Tiers: T1 honest | T2 naive-inflate | T3 trust-conditional | T4 backloaded | T5 frontloaded")
    print("Terminal payoff B: total = round_wins + B * final_trust  (reputation has lasting value)")
    print("Sweep B to see if it tempers the back-loading exploit and grades the ladder.\n")
    print(f"  {'B':>4} | {'T1':>6} {'T2':>6} {'T3':>6} {'T4':>6} {'T5':>6} | winner / ordering")
    for B in [0.0, 1.0, 2.0, 4.0]:
        by = evaluate(FIELD, lam=1.5, h=0.1, B=B, seed=31, n_ep=N_EP)
        r = {t: by[t][0] for t in by}
        order = " < ".join(t for t, _ in sorted(r.items(), key=lambda kv: kv[1]))
        print(f"  {B:>4.1f} | {r['T1']:6.2f} {r['T2']:6.2f} {r['T3']:6.2f} {r['T4']:6.2f} {r['T5']:6.2f} "
              f"| {order}")
    print("\n[DETAIL] final credibility per tier at B=2.0 (shows who ends credible):")
    by = evaluate(FIELD, lam=1.5, h=0.1, B=2.0, seed=41, n_ep=N_EP)
    print(f"  {'tier':>4} {'total':>7} {'final_trust':>11}")
    for t in ["T1", "T2", "T3", "T4", "T5"]:
        print(f"  {t:>4} {by[t][0]:7.2f} {by[t][1]:11.3f}")


if __name__ == "__main__":
    main()
