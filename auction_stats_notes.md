# Auction fidelity — statistics notes (PIN for Experimental Results writeup)

Captured during analysis; fold into the Mimic Fidelity results when writing.
All numbers from the RECENT data: real = folder_19 (save_slots 840-849, 10 auctions / 120 paintings),
mimic = folder_20 (save_slots 850-879, 30 auctions / 360 paintings).

## 1. Unit of analysis: report n in AUCTIONS, not paintings
- n = **10 real / 30 mimic auctions** is the independent replication count. Paintings within an
  auction are NOT independent (shared bidders, depleting budgets, one coupled trajectory; 12 wins
  always split among 5 bidders).
- Do NOT report "n = 120 / 360 paintings" as the sample size — that's the wrong unit.
- More AUCTIONS (not more paintings) is the lever for power/diversity.

## 2. The auctions are near-deterministic (key empirical fact)
Per-auction win counts (out of 12) are almost constant across the 10 real auctions:
GPT ~2, Grok ~2, Llama 3-4, Opus ~2, Pro 2-3.
- Between-auction SD: 0.32 / 0.32 / 0.42 / 0.42 / 0.52
- SD if paintings were iid-binomial: 1.32 / 1.26 / 1.53 / 1.34 / 1.43  (i.e. ~4x larger)
- Design effect ~0.1 (clustering REDUCES variance; negative intra-auction correlation from the
  compositional constraint + consistent strategies).
- CONSEQUENCE: the painting-level chi-squared is CONSERVATIVE here, NOT anti-conservative.
  (Earlier worry about pseudoreplication inflating confidence does NOT apply — it's the opposite.)
- One-sentence finding worth stating: "the LLMs converge on a near-constant win allocation across
  repeated auctions."

## 3. Numbers to report (already computed — no need to rerun)
Outcome-level fidelity, real vs mimic win shares (pp = percentage points):
  Model     real%  mimic%   Δ(mimic-real)   auction-level 95% CI
  GPT-5.4   17.5   18.3     +0.8            [-1.4, +2.5]
  Grok      15.8   14.2     -1.7            [-3.6, +0.6]
  Llama     26.7   28.6     +1.9            [-0.6, +4.2]
  Opus      18.3   20.0     +1.7            [-0.8, +4.2]
  Pro       21.7   18.9     -2.8            [-6.1, +0.6]
- Cramér's V = 0.041, 95% CI [0.020, 0.079]  (upper bound below the 0.1 "negligible" threshold)
- Chi-squared independence: chi2 = 0.81, p = 0.94  (KEEP as a secondary sanity line only)
- JS divergence = 0.0016 bits, JS distance = 0.040  -> DO NOT report: it ~equals Cramér's V (0.041),
  redundant, adds no information and no significance.

## 4. Recommended framing for the fidelity paragraph
- LEAD with effect-size + CI / equivalence: "Cramér's V = 0.041 (95% CI [0.02, 0.08], below the 0.1
  negligible threshold); all per-model win-share differences within ~6 pp (95% CI)."
- DEMOTE chi-squared to one secondary sanity-check sentence (don't lean on p=0.94; absence-of-
  evidence != evidence-of-equivalence).
- KEEP the action-level CV table (per-decision raise/pass 68-85% acc) as the complementary level.
- KEEP the jump-bid limitation (mimics make ~0 jump bids; end-game spike 2.4x vs 3.8x real).
- Optional: pre-register an equivalence margin (TOST) to turn the CI into a formal equivalence claim
  (limited by n=10: Pro's CI brushes +-6 pp).

## 5. Genuine limitation to state (not pseudoreplication)
The 10 auctions are near-identical scenarios (12 identical paintings, equal budgets; only seat order
+ LLM sampling vary). So they give precision but little DIVERSITY. If adding runs, vary conditions
(budgets, # bidders, non-identical item values) rather than running more near-copies.

## 5b. v6/v7 LINEAGE BUG (fix before publishing the action-level table)
- Deployed mimics = models/v7/*.pt, written 2026-05-24 20:04-20:12, trained on the v7 dataset
  (auction_nn_dataset_v7, created 20:01) = the 10 recent auctions (save_slots 840-849).
- v7 training set = **3,334 decisions** (GPT 697, Grok 628, Llama 670, Opus 672, Pro 667).
  -> Prose now says 3,334 (fixed). NOT 1,700.
- The action-level fidelity TABLE currently in the tex (decisions 434/310/312/465/174 = 1,695;
  CV acc GPT .84 / Grok .77 / Llama .69 / Opus .85 / Pro .68; range "68-85%") is from the OLD
  **v6** run (last_training.log, dated May 10), NOT the deployed mimics. Those values are WRONG
  for the paper.
- RESOLVED: re-ran LOO-CV on the v7 dataset (log: training_logs/v7_cv_rerun.log; checkpoints to
  models/v7_cv_rerun, deployed models/v7 untouched). v7 numbers are MUCH better than v6:
    Model     Decisions  CV raise/pass acc   Overbid MAE (steps)
    GPT-5.4   697        0.93 +/- 0.02       0.46
    Grok      628        0.89 +/- 0.03       0.02
    Llama     670        0.88 +/- 0.02       0.44
    Opus      672        0.90 +/- 0.05       0.51
    Pro       667        0.91 +/- 0.03       1.55
  -> Action-level (decision-level) table RE-ADDED to the tex with these numbers (88-93% acc).
     Pro's MAE 1.55 ties to the jump-bid limitation. Reversed the earlier "drop the table" call.
  - Caveat: these are from a fresh CV run (torch RNG was never seeded), so they characterize the
    v7 training PROCEDURE, not the exact deployed weights bit-for-bit (the deployed mimics are one
    instance of the same procedure on the same data).
- (Checkpoints are confusingly named auction_*_v6_*.pt but live in models/v7/ and ARE the v7-trained
  models -- the "v6" in the filename is a stale prefix, not the data version.)

## 6. Separate open item (decision-validity / transfer)
Mimic-mediated baseline results (Trivial/Fair-Share/Reactive/Market-Clearing/RL) are validated vs
mimics only. Only RL was checked vs real LLMs (folder_30, 5 auctions, 36.7%). To claim transfer,
run each heuristic vs real LLMs and compare the RANKING (rank correlation), not absolute rates.
If not run, scope the claim to "vs the mimic ensemble" and cite the jump-bid limitation as the
reason transfer is uncertain. (Do NOT skip the experiment to avoid the result.)
