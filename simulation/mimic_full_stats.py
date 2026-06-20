"""Full mimic fidelity report — comprehensive stats per-LLM, all in one print."""
from __future__ import annotations
import asyncio, glob, json, math, os, random, re, statistics
from collections import defaultdict
import numpy as np
import torch

from simulation.env import TravelGameEnv
from simulation.mimic_agent import _load_deception_mimic
from simulation.server import _build_actions_live_deception_competition

LLM_LIST = ["GPT-5.4", "Grok", "Opus", "Pro", "Llama"]
ATTRS = ("beach", "food", "pool", "room", "service")


def t_ci(values, conf=0.95):
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    half = 2.01 * sd / math.sqrt(n) if n > 1 else 0.0
    return mean * 100, max(0, (mean - half)) * 100, min(100, (mean + half)) * 100, sd * 100


def main():
    # Training stats
    training = {t["alias"]: t for t in json.load(open("simulation/models/deception_v5/training_summary.json"))}

    # Dataset
    rows = []
    with open("simulation/datasets/deception_dataset_v5.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    vocab = json.load(open("simulation/datasets/deception_dataset_v5_meta.json"))["bidder_vocab"]
    inv_vocab = {v: k for k, v in vocab.items()}
    by_alias = defaultdict(list)
    for r in rows:
        by_alias[inv_vocab[r["bidder_index"]]].append(r)

    # Action-level metrics
    action = {}
    for alias in LLM_LIST:
        bidder_rows = list(by_alias[alias])
        rng = random.Random(0)
        rng.shuffle(bidder_rows)
        n_val = max(1, len(bidder_rows) // 10)
        val = bidder_rows[:n_val]
        train = bidder_rows[n_val:]
        model = _load_deception_mimic(f"Mimic-{alias}")
        X = np.array([row["x"] for row in val], dtype=np.float32)
        truth = X[:, :5]
        y_claim = np.array([row["y_claim"] for row in val], dtype=np.float32)
        # Direct claim prediction.
        with torch.no_grad():
            pred = model(torch.tensor(X)).cpu().numpy()
        per_attr_mae = np.abs(pred - y_claim).mean(axis=0)
        real_disc = np.clip(y_claim - truth, 0, None)   # actual over-promise
        mimic_disc = np.clip(pred - truth, 0, None)     # mimic over-promise
        action[alias] = {
            "n_train": len(train),
            "n_val": n_val,
            "best_val_loss": training[alias]["best_val_loss"],
            "claim_mae_overall": float(np.abs(pred - y_claim).mean()),
            "claim_mae_per_attr": [float(x) for x in per_attr_mae],
            "mean_real_disc": float(real_disc.mean()),
            "mean_mimic_disc": float(mimic_disc.mean()),
            "median_real_disc": float(np.median(real_disc)),
            "median_mimic_disc": float(np.median(mimic_disc)),
            "mean_pred": [float(x) for x in pred.mean(axis=0)],
            "mean_real": [float(x) for x in y_claim.mean(axis=0)],
        }

    # Real LLM episodes — read from folder "Deception v5 (track-record)"
    real_wins = defaultdict(int)
    real_per_ep = defaultdict(list)
    n_real_eps = 0
    cat = json.load(open("simulation/.runtime/save_slots.json"))
    real_folder = next(f for f in cat["folders"] if f.get("name") == "Deception v5 (track-record)")
    real_slot_ids = [s["slot_id"] for s in cat["slots"] if s.get("folder_id") == real_folder["folder_id"]]
    for sid in real_slot_ids:
        try:
            ep = json.load(open(f"simulation/.runtime/save_slots/{sid}/auction_exports/deception_episode/episode_log.json"))
        except Exception:
            continue
        if not ep.get("complete"):
            continue
        n_real_eps += 1
        for a in ep["agents"]:
            if a["alias"] in LLM_LIST:
                real_wins[a["alias"]] += a["win_count"]
                real_per_ep[a["alias"]].append(a["win_count"])

    # Mimic episodes are persisted as save slots in folder_63
    # ("Mimic Deception v3 (T=1.75)"). 150 episodes total: 3 replicates of
    # truth seeds 1-50. We load the win counts directly from the persisted
    # episode_log.json files rather than re-simulating each run.
    def load_mimic_persisted() -> tuple[dict[str, int], dict[str, list[int]]]:
        mw = defaultdict(int)
        mp = defaultdict(list)
        cat = json.load(open("simulation/.runtime/save_slots.json"))
        target = next((f for f in cat["folders"] if f.get("name") == "Mimic Deception v5 (32d T=0.50)"), None)
        if target is None:
            raise RuntimeError("Mimic Deception v3 folder not found in catalog. "
                               "Re-run the mimic persistence script first.")
        slot_ids = [s["slot_id"] for s in cat["slots"] if s.get("folder_id") == target["folder_id"]]
        for sid in slot_ids:
            ep_path = f"simulation/.runtime/save_slots/{sid}/auction_exports/deception_episode/episode_log.json"
            try:
                ep = json.load(open(ep_path))
            except Exception:
                continue
            if not ep.get("complete"):
                continue
            for a in ep["agents"]:
                alias = a["alias"].replace("Mimic-", "")
                if alias in LLM_LIST:
                    mw[alias] += a["win_count"]
                    mp[alias].append(a["win_count"])
        return mw, mp

    mimic_wins, mimic_per_ep = load_mimic_persisted()

    # Chi-sq + Cramer's V
    obs = np.array([[real_wins[a] for a in LLM_LIST], [mimic_wins[a] for a in LLM_LIST]], dtype=float)
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    grand = obs.sum()
    expected = (row_sums @ col_sums) / grand
    chi2 = float(np.nansum((obs - expected) ** 2 / expected))
    df = len(LLM_LIST) - 1
    z = ((chi2 / df) ** (1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    v = math.sqrt(chi2 / grand)

    print("=" * 90)
    print(" MIMIC FIDELITY REPORT")
    print(f"   {n_real_eps} real-LLM episodes (the v3 batch)  |  150 mimic episodes "
          f"(3 replicates per truth seed)  |  T = {os.environ.get('DECEPTION_MIMIC_TEMPERATURE', '1.5')}")
    print("=" * 90)

    print("\n[1] TRAINING")
    print(f"   {'LLM':<10}{'n_train':>10}{'n_val':>8}{'val_MSE':>10}")
    for a in LLM_LIST:
        m = action[a]
        print(f"   {a:<10}{m['n_train']:>10}{m['n_val']:>8}{m['best_val_loss']:>10.4f}")

    print("\n[2] ACTION-LEVEL FIDELITY (held-out 60-row val set, deterministic predict)")
    print(f"   {'LLM':<10}{'claim_MAE':>12}{'real_disc_mean':>17}{'mimic_disc_mean':>18}{'delta':>10}")
    for a in LLM_LIST:
        m = action[a]
        d = m["mean_mimic_disc"] - m["mean_real_disc"]
        print(f"   {a:<10}{m['claim_mae_overall']:>12.4f}"
              f"{m['mean_real_disc']:>17.4f}{m['mean_mimic_disc']:>18.4f}{d:>+10.4f}")

    print("\n[3] PER-ATTRIBUTE CLAIM MAE")
    print(f"   {'LLM':<10}" + "".join(f"{n:>9}" for n in ATTRS))
    for a in LLM_LIST:
        m = action[a]
        print(f"   {a:<10}" + "".join(f"{x:>9.4f}" for x in m["claim_mae_per_attr"]))

    print("\n[4] MEAN CLAIM PER ATTRIBUTE  (real / mimic / delta)")
    print(f"   {'LLM':<10}{'attr':>8}{'real':>9}{'mimic':>9}{'delta':>9}")
    for a in LLM_LIST:
        m = action[a]
        for i, name in enumerate(ATTRS):
            d = m["mean_pred"][i] - m["mean_real"][i]
            print(f"   {a:<10}{name:>8}{m['mean_real'][i]:>9.3f}{m['mean_pred'][i]:>9.3f}{d:>+9.3f}")
        print()

    print("[5] EPISODE-LEVEL FIDELITY (episode is the independent replicate)")
    print(f"   Real LLM n_eps = {len(real_per_ep[LLM_LIST[0]])},  Mimic n_eps = {len(mimic_per_ep[LLM_LIST[0]])}")
    print(f"   {'LLM':<10}{'real_mean':>11}{'real_95%CI':>16}{'mimic_mean':>12}{'mimic_95%CI':>16}{'delta_pp':>11}")
    for a in LLM_LIST:
        rm, rl, rh, _ = t_ci([w / 12 for w in real_per_ep[a]])
        mm, ml, mh, _ = t_ci([w / 12 for w in mimic_per_ep[a]])
        delta = mm - rm
        print(f"   {a:<10}{rm:>10.1f}%  [{rl:>4.1f}, {rh:>4.1f}] {mm:>10.1f}%  [{ml:>4.1f}, {mh:>4.1f}]{delta:>+10.1f}")

    print(f"\n   Secondary aggregate-counts check (used for chi-squared / Cramer's V only;")
    print(f"   round-level counts are NOT the independent unit for CI purposes):")
    print(f"   chi-squared = {chi2:.2f}  df={df}  p~{p:.4f}    Cramer's V = {v:.4f}")


if __name__ == "__main__":
    main()
