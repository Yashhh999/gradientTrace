# CHANGELOG — gradientTrace run log

Append-only dated record of what was tried and what the gate said. This is the archive;
`AGENTS.md` is the operating manual and stays current. Newest entries first.

Measurement rules live in AGENTS.md §8, the acceptance gate and its baselines in §7.5, and the
hybrid's architecture and ablations in §7.8. Nothing here is a to-do: if an entry contradicts
AGENTS.md, AGENTS.md is right and this one is stale.

---

## 2026-11-04

- **Fused gate submission (2026-11-04).** `colab/external_gate.py` gained a `FUSION_DIR` knob so a
  saved fusion can go through the exact acceptance harness, plus a restore path that recovers the
  frozen dataset from the Hub whenever it is missing (previously restoring only on push, which sent
  a local PUSH=False run into regeneration on CPU). Two runs: the gate-fitted artifact
  (`v2-pilot-stylo`) scored pooled AUROC 0.9759 / TPR@1% 0.6000 / FPR@0.5 0.0862 and pairwise 0.9900,
  which reproduces the in-bank numbers, so it is an in-bank measurement and not a deployment claim.
  The artifact this project will ship (`v2-pilot-stylo-ext`, fit on held-out slices only) scored
  news 0.9491 / 0.8175 / 0.0800 and reviews 0.9498 / 0.3800 / 0.0425, pooled 0.9421 / 0.5850 /
  0.0612, pairwise 0.9600, against the pilot's 0.9487 / 0.7925, 0.9360 / 0.3650, 0.9323 /
  0.5887. Fitted on the bank and validated by the gate cell, both sets now clear all four bars.

- **Stylometry pruning, mean pooling and the k sweep (2026-11-04).** Dropping `log_words`,
  `sent_mean_words` and `llm_marker_rate` each improved the out-of-bank gate result slightly (AUROC
  0.9414 -> 0.9421, bank-threshold catches 458 -> 464 of 800), so all three are removed from
  `STYLO_NAMES` and the shipped artifact regenerated; `llm_marker_rate` fired on only 2 of the 20
  frontier texts it was designed to catch. Mean pooling lifts the k-NN block from 0.9102 to 0.9277
  pooled AUROC at k=8 (k=16: 0.9302, k=32: 0.9317) and reviews TPR@1% from 0.2375 to as much as
  0.3425, but it still never beats the detector (0.9324 pooled, 0.3850 reviews), so geometry stays
  diagnostic and out of the shipped set. Temperature scaling fitted on the bank's out-of-fold logit
  gave T=1.7352 and changed FPR@0.5 from 0.1925 to 0.1925 — a monotone transform cannot change
  AUROC or a same-data threshold — so it is documented as a measured no-op and kept out of the path.
- **Chunked scoring (2026-11-04).**
- **Routing, abstention, pruning, temperature: measured out-of-bank (2026-11-04).**
  `colab/hybrid_routing.py` runs entirely from the bank/gate caches and reports three things. A
  cost-weighted policy (false accusation = 10, miss = 1) learned on the bank rows where the systems'
  costs differ improves news to TPR@1% 0.8225 (fusion 0.8025, detector 0.7925) and the human FPR to
  0.0013 (fusion 0.0100, detector 0.0063) — but it is trained on only 99 informative bank rows, so
  the effect is small and the abstention counts are the substantive finding: abstaining on the
  46/1,600 gate rows where detector and fusion disagree (39 AI, 7 human) leaves both systems at
  425/761 AI caught with 3/793 false accusations on the scored rows. The off-register probe shows
  the fusion *suppressing* explainer scores (ai-explainer 0.0411 -> 0.0034, product copy 0.5255 ->
  0.0649) while Wikipedia's GBR lead stays at 0.9357, above the 0.8730 operating cut and still
  flagged; coverage is not improved by any of this. Pruning is a small consistent win:
  `log_words`, `sent_mean_words` and `llm_marker_rate` each reduce the result when removed
  (full AUROC 0.9414 / 458 caught; without all three 0.9421 / 464), so they are queued for removal
  from STYLO_NAMES but the effect is below the bank's noise floor. Temperature scaling fitted on
  the bank's out-of-fold logit gave T=1.7352 and changed FPR@0.5 from 0.1925 to 0.1925 — a measured
  no-op, because a monotone transform cannot change AUROC or a threshold that is calibrated on the
  same data. Chunked re-embeddings moved §7.9 by at most 0.002.

- **Public detector baselines on our frozen gate (2026-11-04).** `colab/gate_baselines.py` scores
  publicly available checkpoints on `external_gate_v1` with our metric. Each checkpoint's label
  convention is read from its own `config.id2label` (OpenAI's is `0=fake, 1=real` — the opposite of
  this repo's `1 = AI`), and checkpoints shipping generic labels are scored in both directions with
  the AI class identified by which orientation gives AUROC > 0.5. Results (pooled / news / reviews,
  TPR at a human-calibrated 1% FPR): OpenAI roberta-large 0.0488 / 0.1400 / 0.0250 (AUROC 0.8220);
  OpenAI roberta-base 0.0288 / 0.0500 / 0.0300 (0.7891); fakespot roberta-base 0.3200 / 0.3175 /
  0.3950 (0.8451); `tabularisai/ai-text-detection` (ModernBERT-base, RAID+6 corpora) 0.3825 /
  0.1800 / 0.6775 (AUROC 0.9386); gradientTrace pilot 0.5887 / 0.7925 / 0.3650 (0.9323); shipped
  fusion 0.5787 / 0.8050 / 0.3700 (0.9422). `shunya510hi/gpt4-text-detector` is unusable (0.5161).
  Reading: news — where neither side shares the gate's human source — is ours by 4.5x on TPR@1%.
  Reviews is lost to tabularisai (0.6775) and also to fakespot (0.3950); tabularisai trained on
  RAID, whose review corpus is Yelp, i.e. the gate's own human reviews, so its number is inflated —
  but our reviews weakness is independently real. tabularisai claims RAID TPR@1% 0.9306 and manages
  0.3825 here, so our gate is harder than RAID for it (RAID's AI side includes easier/older
  generators). Conclusion recorded: further scorer/fusion work has a low measured ceiling; the
  remaining levers are review-domain and register diversity in training data, a second backbone
  A/B, and a gate built on a second generator.



- **Truncation audit and chunked scoring (2026-11-04).** Tokenising the frozen gate with the
  release tokenizer shows only 23 of 1,600 rows exceed 512 tokens (all human Yelp reviews; max
  1,162, p99 555) and no news row does, so the gate cannot detect head-only scoring even though
  it is a real deployment hazard. `predict_probs_and_embeddings` now folds texts into contiguous
  windows and aggregates by token count (`chunk=False` keeps the old behaviour), with a unit test
  that proves the tail is forwarded by counting tokens rather than by comparing probabilities.
  Consequence: every embedding cache created before this is pre-chunk and must be invalidated
  before reuse; the §7.9 tables predate it and move only slightly.
- **Routing and abstention experiment, pruning and temperature tests (in progress).** Same caches,
  `colab/hybrid_routing.py`: a small policy learns from bank rows which system (detector vs shipped
  fusion) costs less at the operating threshold, then routes the gate by that policy; disagreement
  rows become an explicit abstention set instead of a forced call. The same script ablates
  `log_words` / `sent_mean_words` / `llm_marker_rate` and fits a temperature on the bank's
  out-of-fold logit. Mean-pooling vs CLS and a paraphrase-hardened gate slice remain open and are
  recorded as the known gaps rather than silently omitted.

## 2026-11-03

- **External-bank re-test refutes the gate-fitted hybrid (2026-11-03).** `colab/hybrid_bank_eval.py`
  refit the fusion on a 4,575-row bank drawn from the pilot's held-out slices (in-domain,
  unseen-generator, humanized, held-out poetry; balanced by slice/label/domain), set the threshold
  from that bank's out-of-fold scores, and scored the frozen gate as an unseen population. The
  gate-fitted result (pooled AUROC 0.9876, TPR@1% 0.7688, FPR@0.5 0.0550, 9/20 frontier explainers
  caught) came apart: every k-NN feature set fell below the bare detector on pooled AUROC
  (`det + vectors` 0.9105, `hybrid all` 0.9136 vs detector 0.9323), reviews TPR@1% dropped
  0.3650 -> 0.2125/0.2400, and explainer coverage returned to 1/20. Stylometry survived and
  improved: `det + stylometry` reached pooled AUROC 0.9422, news AUROC 0.9480 / TPR@1% 0.8050 /
  FPR@0.5 0.0825 (from 0.2475), reviews AUROC 0.9494 / TPR@1% 0.3700 / FPR@0.5 0.0450 (from
  0.1375); at the bank-calibrated 0.8741 cut it caught 459/800 gate AI texts with 8 human false
  alarms versus the detector's 471/800 at its own 1% point. Reading: embedding geometry is
  bank-local, surface statistics are corpus-portable — the opposite of the prediction made before
  running it. Shipped `SHIP_SET="det + stylometry"`, which needs no bank at inference.
- **Hybrid feature work (2026-11-03).** Dropped the GPT-2 reference LM (pooled 0.9481 in-bank, 2/20
  explainers, `bpb` ranking archaic prose as machine text) and replaced it with two signals derived
  from the detector itself: 33 stylometric features and k-NN/centroid geometry over the pooled
  `encode()` vector. Added `gradienttrace/hybrid.py`, `predict_probs_and_embeddings` (bit-identical
  logit to `predict_probs`, asserted), fold-rebuilt vector features with pair-partner blocking
  (asserted in `tests/test_hybrid.py`, now wired into CI), per-fold coefficient sign agreement, and
  a bank-less `HybridScorer` path.

- **Hybrid system (2026-11-03, `colab/hybrid_eval.py`):** fusing the detector logit with GPT-2
  surprisal features by out-of-fold logistic regression halves the human false-positive rate at
  equal detection (news FPR@0.5 0.2475 -> 0.1000, TPR@1% 0.7925 -> 0.7825; reviews TPR@1%
  0.3650 -> 0.4025; pooled AUROC 0.9323 -> 0.9481) and the reference LM alone reaches news AUROC
  0.9226, so the second signal is real. `log_chars` must stay out of the feature set. It does not
  buy coverage: on 20 frontier explainers the detector flags 1/20 and the hybrid 2/20 at each
  system's own 1% FPR point, because surprisal measures stylistic familiarity (Declaration of
  Independence bpb 0.163, Wikipedia 0.70-0.75, AI explainers 0.89-1.27). Coverage now needs
  training data in the encyclopedic/explainer register, not a second scorer. Details in 7.8.

- **Ailow10 experiment (2026-11-03):** `configs/v2_ailow10.yaml` cut `local_jsonl.ai_foundational`
  to 0.10 at 607 steps on the 150k pool (local pool 20,000 human / 9,064 AI). Gate: news 0.9176 /
  0.2825 / 0.2825 (human mean 0.3622, AI mean 0.9082), reviews 0.9560 / 0.5450 / 0.3300, pooled
  0.9371 / 0.4612 / 0.3063. The local-AI share does drive the clean-human spill and FPR@0.5 now
  clears 0.36, but news TPR@1% falls to 0.2825 against a 0.60 bar, so it fails the bar too.
  **The 150k data-mix search is closed; `Yashhh999/gradientTrace-v2-pilot` (ailow) is the release
  model.** Internal slices are unchanged in kind (val 0.9948, OOD 0.9993, poetry 0.9507).

- **150k run and gates (2026-11-02/03, done):** the corpus top-up, the 5,265-step v2 run, a
  1,755-step run and a 607-step run all shipped and all failed the same way — news AUROC
  0.935-0.953 with human mean P(AI) 0.62-0.68 and FPR@0.5 0.60-0.71, versus the pilot's 0.2623
  / 0.2475 at the identical 607-step budget. Steps are not the variable.

- **Ailow pilot gate (2026-11-02):** cutting local AI 0.4 -> 0.25 cleared the pilot bar. Gate
  (`external_gate_v1_report_v2-pilot-ailow.json`, model `-v2-pilot`): news 0.9487 / 0.7925 /
  0.2475 (human mean 0.2623, p50 0.0558, p90 0.9193, p99 0.9922; AI mean 0.9196; threshold
  0.9928), reviews 0.9360 / 0.3650 / 0.1375 (human mean 0.1639, AI 0.8004), pooled 0.9323 /
  0.5887 / 0.1925 (pair 0.9575, matched 0.9053). Human data was identical to longhuman, so the
  whole move is the weaker journalist-prompt/short-AI contrast: clean-human mean 0.4273 ->
  0.2623 in both length buckets (0–400 0.432 -> 0.263, 400–800 0.310 -> 0.268), TPR@1% +0.175,
  AUROC +0.016. Reviews TPR@1% 0.365 is below the ~0.40 watch line but Yelp is source-seen.
  Internal val 0.9961 / 0.948, in-domain 0.9959 / 0.909, OOD 0.9990 / 0.982, humanized 0.9998 /
  0.991, poetry 0.9624 / 0.538.

- **Longhuman2 gate (2026-11-02):** raising the local human weight to 1.0 was a regression.
  Gate (`external_gate_v1_report_v2-pilot-longhuman2.json`, model `-v2-pilot`): news 0.9299 /
  0.4825 / 0.5325 (human mean 0.5431, p50 0.6039, p90 0.9957; AI mean 0.9699; threshold 0.9987),
  reviews 0.9697 / 0.5450 / 0.1700 (human mean 0.1949, AI 0.9377), pooled 0.9278 / 0.4412 /
  0.3513. AUROC stayed flat while both means rose: the larger local news-AI contrast sharpened
  AI scores and spilled onto clean humans, and the news human distribution shifted up in both
  length buckets (0–400 0.432 -> 0.546, 400–800 0.310 -> 0.495). The 0.5 human mix remains the
  best news model.

- **Longhuman pilot gate (2026-11-02):** the builder prune worked and this is the best news
  model so far. Corpus 10,000 long humans / 15,002 AI (7,002 short); local quota 6,667 rows,
  all long (train ~5,333). Internal val 0.9942 / 0.867 (threshold 0.867), in-domain 0.9944 /
  0.895, OOD 0.9984 / 0.961, humanized 0.9995 / 0.981, poetry 0.9578 / 0.516. Gate: news
  0.9323 / 0.6175 / 0.4225 (human mean 0.4273, AI 0.9406; 1% threshold 0.9978 = human p99),
  reviews 0.9151 / 0.3125 / 0.2275 (human mean 0.2425), pooled 0.9088 / 0.4412 / 0.3250 (pair
  0.9525, matched 0.8786). TPR@1% clears 0.5 for the first time, but FPR@0.5 is still 0.06
  above the bar and the human residual is in 0–400 chars (0.4323 vs 0.3103 at 400–800).

- **Diagnosis (2026-11-02):** the local human quota is fixed by `source_shares` (6,667 rows at
  the pilot) and the pool is shuffled, so every short human row displaces a long one. Long local
  train fell ~3,800 (short pilot) -> ~2,900 (balanced), and the gate's humans are 247-char
  headline+wire. The long-only cc_news model (5,333 long local humans) had the best calibration
  (human mean 0.3746) and even scored the same 0–400 gate humans well, so long wires generalise
  to the gate's short/medium texts better than short human rows do. The short pilot remains the
  best news model (0.9108 AUROC, 0.5400 TPR@1%, human mean 0.5095).

- **Valid short2 gate (2026-11-01):** the balanced build made human calibration worse. Gate
  (`external_gate_v1_report_v2-pilot-short2.json`, model `Yashhh999/gradientTrace-v2-pilot`):
  news 0.8994 / 0.4000 / 0.5925 (human mean 0.5095 -> 0.5926, p50 0.4824 -> 0.6754, p90 0.9920,
  p99 0.9983; AI mean 0.9359 -> 0.9422), reviews 0.9585 / 0.5200 / 0.1150 (mean human 0.1560),
  pooled 0.8937 / 0.3862 / 0.3538 (pair 0.9437, matched 0.8624). Both the 0–400 human bucket
  (0.514 -> 0.594) and 400–800 (0.418 -> 0.592) moved up, so the cue is not short-text-specific.

- **Balanced-shorts pilot (2026-11-01):** corpus 18,589 human / 14,999 AI (8,589 short humans =
  6,000 cc_news + 2,589 paired CNN; 7,002 short AI), local 10,481 train rows, train 38,571.
  Internal val 0.9960 / 0.933, in-domain 0.9960 / 0.928, OOD 0.9992 / 0.984, humanized 0.9999 /
  0.997, poetry 0.9651 / 0.476. The first gate attempt left `MODEL_ID` at the v1 repo; that
  report was overwritten by the valid rerun below, and the cell now aborts on the mistake.

- **Cause (2026-11-01):** short AI rows were 47% of the AI corpus but short humans only 29%,
  and every short human was an unpaired cc_news brief while every short AI text had a long
  parent. The model learned "short news = AI" on top of the form fix. Length is still not the
  only cue (news AUROC 0.91, matched AUROC 0.88), but the human score distribution shifted up
  as a whole (p50 0.25 -> 0.48), which fits a length threshold near the gate's 247-char median.

- **Short-variant pilot (2026-11-01):** the AI side moved and the bottleneck is human
  calibration. Corpus 14,000 human / 14,984 AI (7,001 short AI), local 10,512 train rows, train
  news 14,017 of 38,636 (36%). Internal val 0.9960 / 0.933, in-domain 0.9956 / 0.931, OOD
  0.9992 / 0.983, humanized 0.9998 / 0.994, poetry 0.9593 / 0.470 (down from 0.9713 / 0.623).
  Gate (`external_gate_v1_report_v2-pilot-short.json`): news 0.9108 / 0.5400 / 0.4975 (human
  mean 0.3746 -> 0.5095, p50 0.2457 -> 0.4824; AI mean 0.7876 -> 0.9359; 1% threshold 0.9968 =
  human p99), reviews 0.9526 / 0.4950 / 0.1175 (better), pooled 0.9032 / 0.4263 / 0.3075 (pair
  0.9463, matched 0.8761). News AUROC +0.10 and TPR@1% FPR +0.31, but FPR@0.5 +0.14. **Not
  shippable**; the model now ranks news well but calls half the clean humans AI at 0.5.

- **Length/form diagnosis (2026-11-01):** the cc_news pilot's AI failure is a distribution gap,
  not a generator failure. Gate Qwen news is 1.3 sentences (chars p5/p50/p95 = 78/134/223;
  first sentence p50 122). Training AI news is 3.8 sentences (chars 220/331/457) because
  `min_new_tokens` forced gate-length output and none of the 7,448 rows is shorter than ~220.
  The classes had zero length overlap, so short clean text read as human; a model that never
  saw a short AI news brief cannot score one. Content was ruled out: content-word recall of the
  gate's topic is 0.41 (Qwen) vs 0.44 (Phi) / 0.54 (SmolLM), and first-sentence lengths already
  match the gate (p50 124), so truncation reproduces the gate shape (sampled short AI p50 135).

- **cc_news pilot (2026-11-01):** the human-form fix works and the bottleneck moved to the AI
  side. `v2/news_pairs.jsonl` grew to 10,000 human rows (4,000 CNN leads + 6,000 cc_news) plus
  7,448 AI texts, and `local_jsonl.human` rose to 0.5, so local_jsonl supplied 9,038 train rows
  (human bucket 16,064, local ~5k = 31%; the extra
  weight displaced hc3/mage/raid, not the bucket size). Internal val 0.9962 / 0.933, in-domain
  0.9957 / 0.911, OOD 0.9992 / 0.985, humanized 0.9991 / 0.988, poetry 0.9713 / 0.623. Gate
  (`external_gate_v1_report_v2-pilot-ccnews.json`): news 0.8131 / 0.2325 / 0.3575 (human mean
  P(AI) 0.9180 -> 0.3746, human p50 0.9942 -> 0.2457; AI mean 0.9653 -> 0.7876), reviews 0.9293
  / 0.3725 / 0.1500, pooled 0.8725 / 0.2637 / 0.2537. News AUROC +0.22 and FPR@0.5 -0.57; the
  headline+wire form from many outlets was the human-side gap. The AI mean fell 0.18 and only
  23% of Qwen news is caught at 1% FPR — the 1% threshold sits at human p99 = 0.9958.

- **Diagnosis (2026-11-01):** in-domain news is solved (0.998), so this is a source/form shift,
  not a news failure. The gate's ag_news humans are a headline followed by a wire description
  (p50 247 chars); training human news is CNN article leads with no headline (p50 248, so not
  length) plus perturbed RAID BBC news. Only 20% of the human bucket is clean news, and more
  CNN repeats the same source. The model has never seen the gate's form from many outlets.

- **Clean-news pilot (2026-11-01):** the first run whose mix contains the corpus: local_jsonl
  6,903 train rows (3,224 human + 3,679 AI), human bucket 16,056, so 20% clean news. Internal
  val 0.9971 / 0.953, in-domain 0.9970 / 0.940, OOD 0.9994 / 0.988, humanized 0.9997 / 0.994,
  poetry 0.9652 / 0.583; `domain:news` is 0.998 on val and in-domain. Gate
  (`external_gate_v1_report_v2-pilot-news.json`): news 0.5906 / 0.0350 / 0.9300 (human mean
  P(AI) 0.9180, AI mean 0.9653), reviews 0.9429 / 0.4000 / 0.1925, pooled 0.7319 / 0.0963 /
  0.5613. The gain is AI-side (0.9318 -> 0.9653; part of it is recognition of the gate's own
  journalist prompt, which the training AI news reused). The human side did not move:
  0.9208 -> 0.9180 and 93% still above 0.5. **Does not clear the gate; do not start the 150k.**

- **Clean-news builder (2026-11-01):** complete. `v2/news_pairs.jsonl` holds 4,000 human
  leads + 7,448 AI texts (Phi-3 3,621, SmolLM2 3,827), 3,979 stories covered and 3,469 with
  both generators; no duplicates or sub-200-char rows. Rejections: 486 short, 4 preamble,
  62 no_sentence. AI p50 is 337 chars vs human 248 (capped at 1.6×). The first pass kept only
  402 Phi texts before Granite-2B crashed with NaN logits; `min_new_tokens` and the SmolLM2
  swap fixed it. `one_shot_kaggle.py` downloads the pairs into `data/local.jsonl` and refuses
  to train without them; `configs/{v2,pilot}.yaml` give it 0.25 of the human and AI-foundational
  buckets.

- **Clean-human probe (2026-11-01):** the full RAID blocks confirm there are none to recover:
  news 655,048 rows (8 attacks × 1,780 humans, 0 clean, 39,160 clean AI), reddit 625,221
  (0 clean), books 591,329 (0 clean). Poetry hit Python's csv field limit (131,072) and the
  probe stopped. The extractor's caps were not why the slice looked clean-less; RAID train
  simply has no clean human news to give.

- **Pilot data build (fixed, 2026-11-01):** train 38,717 / val 5,208; buckets 16,052 human /
  15,963 foundational / 6,702 frontier; slices in-domain 4,520, OOD 11,948 (+48 from the qwen
  holdout), humanized 756, held-out poetry 5,060. hc3 pools 7,564/3,064, raid 12,064/13,564/
  2,381, arena 6,064 frontier; mage fills its share across the 9 non-yelp domains. Frontier
  supply is short (~8.4k of the 10k pilot target; ~20.4k of 30k at 150k); fix after the gate.

- **Diagnosis (2026-11-01):** the RAID human side is perturbed in news, reddit, books and
  poetry and clean only in reviews/wiki/recipes, while AI rows are clean everywhere. The model
  learned domain-specific "perturbed = human, clean = AI". The gate's clean ag_news humans
  score mean P(AI) 0.92, while clean Yelp humans score 0.16 — reviews works because clean Yelp
  humans are in training, i.e. it is recognising Yelp, not machine text. MAGE's clean humans
  cover Q&A/summary/forum styles, not news articles.

- **Invalid news pilot (2026-11-01):** a pilot was trained and pushed to `-v2-pilot` before
  `build_news.py` ran (`v2/news_pairs.jsonl` is 404 on the Hub). The new config was in effect
  (`source_shares.local_jsonl` 0.25) but the pasted cell was stale and skipped the download,
  so `manifest["sources"]["local_jsonl"]` is 0 and the mix is identical to the fixed pilot
  (train 38,717 / val 5,208; internal val 0.9958). **Do not gate it**; it cannot test the news
  fix. `build_datasets` now aborts on this state.

- **Pilot history:** pre-fix pilot (2026-10-31) ran 607 steps in 4,637 s with val 0.9972.
  Fixed pilot (2026-11-01) ran the same 607 steps in ~80 min: val 0.9966/0.937, in-domain
  0.9966/0.933, OOD 0.9993/0.987, humanized 0.9993/0.984, held-out poetry 0.9732/0.577.

- **External gate v1:** v1 pooled 0.5270 / 0.0000 / 0.9750; first (label-inverted) pilot
  0.5745 / 0.0263 / 0.9675; fixed pilot 0.7258 / 0.1288 / 0.5463, but news alone 0.4775 /
  0.0125 / 0.9700 (§7.5). All unusable. The dataset is frozen and stays the comparison.
