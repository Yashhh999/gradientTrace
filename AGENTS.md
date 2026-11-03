# AGENTS.md — gradientTrace project context

This is the operating manual for any coding agent working in this repository. It records
what the project is, how the pipeline actually behaves, the traps that have already cost
real runs, and the current state of the work. Update this file when behavior changes.

---

## 1. TL;DR

- **Product:** a binary AI-text detector. Input: text (truncated to 512 tokens). Output:
  `sigmoid(logit) = P(machine-generated)`. Labels: `0 = human`, `1 = AI`.
- **Model:** `microsoft/deberta-v3-large` (435M) encoder + `Linear(1024→1)` head, BCEWithLogits.
  Parameter count is not the lever. Do not re-open the model-size or hardware debate.
- **What decides quality:** paired data (same prompt for human and AI), generator-disjoint
  eval, humanized/paraphrased AI, and fresh frontier generators. Not architecture.
- **The internal eval numbers are not detection quality.** The v1 model reports AUROC ≈ 0.9999
  on every internal slice because the slices are source-confounded (domain equals label).
  A probe showed it flags ordinary human news/reviews as AI ~95–99% of the time.
  The only acceptance signal is the external gate (`colab/external_gate.py`).
- **Current phase (2026-11-03):** the ailow 50k pilot is the release model: news AUROC 0.9487,
  TPR@1% FPR 0.7925, FPR@0.5 0.2475, human mean 0.2623, AI mean 0.9196. The 150k search is
  closed: every 150k build flags 60-71% of clean human news at 0.5 and the last probe
  (`v2_ailow10`, local AI 0.10 at 607 steps) shows why the knob is not the lever — it moved the
  human mean 0.616 -> 0.362 and FPR@0.5 0.600 -> 0.2825 but collapsed news TPR@1% 0.665 -> 0.2825.
  The hybrid is shipped but narrowed (`gradienttrace/hybrid.py`, §7.8-7.9): the detector logit plus
  33 stylometric features, fused by out-of-fold logistic regression. Scored on the gate it looked
  like AUROC 0.9876 with 9/20 frontier explainers caught, but that run used the gate as its own
  k-NN bank. Re-tested with a bank drawn from the model's held-out slices (§7.9), the embedding
  geometry collapsed below the bare detector (pooled AUROC 0.9105 vs 0.9323) and the explainer
  catch rate returned to 1/20, while **stylometry held up**: pooled AUROC 0.9422, news TPR@1%
  0.8050, and FPR@0.5 cut from 0.2475 to 0.0825 on news and 0.1375 to 0.0450 on reviews. The
  shipped fusion is detector + stylometry, no bank. Frontier-explainer coverage is unchanged and
  still needs training data in that register — no scorer fixes it.

---

## 2. Non-negotiables

### Secrets

- **Never print, echo, log, or commit `HF_TOKEN`.** A write-scope token was once printed into
  Kaggle notebook output and had to be revoked. The cells print only “present (value not printed)”.
- The token comes only from the environment or the platform secret store:
  Kaggle `Add-ons > Secrets`, Colab key icon, or `HF_TOKEN` in the env.
- Never use `getpass` inside a notebook cell; it blocks the cell.
- Prefer metadata over secrets in diagnostics: `HfApi.whoami()` exposes the user and token
  role; print those, never the token.
- If Hub calls suddenly return 401, assume the token was rotated after a leak and ask for a
  refreshed secret. Do not assume a repo/permission change.

### Git

- Never modify git config. Use per-command identity:
  `git -c user.name='Yashhh999' -c user.email='Yashhh999@users.noreply.github.com' commit ...`
- Pushes hang on this machine unless the system credential manager is cleared first:

  ```bash
  GIT_TERMINAL_PROMPT=0 git -c credential.helper= \
    -c credential.helper='!gh auth git-credential' push origin main
  ```

  The empty `credential.helper=` resets the helper list; without it, Windows GCM is invoked
  first and blocks behind a GUI prompt.
- Never `--no-verify`, never force-push `main`, never amend published commits.
- Commit style: imperative subject that states the *why*, short body if needed. Examples in
  history: “Skip dateline and one-liner snippets when building gate prompts”, “Fix the Hub
  partial-restore call so cross-session resume works”.

### Files

- Never commit checkpoints, `runs/`, `data/*.jsonl` (except `data/example_local.jsonl`),
  tokens, or secrets. `.gitignore` covers `runs/`, `data/*.jsonl`, weights.
- `colab/*.py` files are pasted whole into one notebook cell. They must stay valid Python,
  import only after dependency checks, and never rely on notebook state.
- The Kaggle/Colab cells clone/pull `https://github.com/Yashhh999/gradientTrace`, so **a code
  change has no effect on a run until it is pushed**.

### Scope

- Do not add features beyond the task. Do not add fallbacks for states the code cannot reach.
- Comments: only for non-obvious *why*. No narration of what the code does.
- Keep `python -m compileall -q gradienttrace colab tests` clean; CI runs it plus
  `python tests/test_metrics.py`.

---

## 3. Repository map

```
gradienttrace/
  config.py     dataclass config schema + YAML loader (rejects unknown keys)
  runtime.py    device/precision/batch detection: tpu | xla | cuda | cpu
  data.py       sources, quotas, dedup, held-out routing, splits, smoke data
  model.py      AIDetector (PreTrainedModel, AutoConfig) + tokenizer helper
  metrics.py    AUROC/AUPRC/TPR@FPR/ECE, sliced reports
  hybrid.py     stylometry + embedding k-NN features, out-of-fold fusion, HybridScorer
  train.py      HF Trainer pipeline, resume, Hub checkpoints, final slice eval
  evaluate.py   standalone sliced evaluation, predict_probs, predict_probs_and_embeddings
configs/
  base.yaml     v1 recipe (hc3 + pile + mage + arena)
  v2.yaml       paired recipe + RAID + held-out domain poetry
  pilot.yaml    50k / 1 epoch pilot of the v2 recipe
colab/
  one_shot.py         Colab cell (GPU or TPU v5e-1), base config only
  one_shot_kaggle.py  Kaggle 2x T4 cell (CONFIG knob, RAID download, Hub preflight)
  external_gate.py    external acceptance gate (frozen matched human/AI dataset)
  finish_eval.py      resumable full slice evaluation on a single GPU
  hybrid_eval.py      hybrid fusion scored on the frozen gate by oof CV; saves fusion.json + bank
  hybrid_bank_eval.py same fusion, bank drawn from held-out slices instead of the gate
  build_raid.py       one-time RAID byte-range extractor -> Hub v2/raid_v2.jsonl
  build_news.py       paired clean-news builder -> Hub v2/news_pairs.jsonl (v2 local.jsonl)
data/example_local.jsonl  schema example for prompt-paired `local.jsonl`
tests/test_metrics.py     metric unit tests
tests/test_hybrid.py      stylometry + fold-safety tests (numpy/sklearn only)
tests/test_hybrid_inference.py  embedding extraction and saved-fusion round trip (needs torch)
.github/workflows/ci.yml  compileall + metric tests
CHANGELOG.md    append-only dated run log (all pilot and 150k gates, diagnoses, dead ends)
```

---

## 4. Model and training path

`AIDetector` (`gradienttrace/model.py`):

- `config_class = AutoConfig`; backbone loaded via `AutoModel.from_config` then weights copied
  from `AutoModel.from_pretrained`. Swapping backbone family is a config change.
- Pooling: `cls` (default) or `mean`; dropout before the linear head.
- `forward` returns `SequenceClassifierOutput(loss, logits)`; loss is BCEWithLogits.
- `gradient_checkpointing_enable()` delegates to the encoder.

`gradienttrace/train.py` startup order matters for diagnosing failures:

1. Parse args → `load_config` (unknown YAML keys raise `KeyError` here).
2. Resolve runtime (`detect_runtime`), apply CLI overrides (`--batch-size`, `--grad-accum`).
3. Resolve Hub id / `push`; `login(token=...)`; `HfApi.create_repo(exist_ok=True)`.
   All DDP ranks call `create_repo`; the resulting 409 is harmless.
4. `build_datasets` → log train/val/slice sizes and the manifest.
5. Tokenize train and val (`pad_to_max` on TPU/XLA).
6. Build model, enable gradient checkpointing, disable `use_cache`.
7. `TrainingArguments` with runtime-aware strategy compatibility:
   `eval_strategy` vs `evaluation_strategy`, `warmup_ratio` vs `warmup_steps`,
   `processing_class` vs `tokenizer` are chosen from the installed signature.
   `hub_strategy` falls back through `[cfg, every_save, end, None]` on `TypeError/ValueError`.
8. `Trainer.train(resume_from_checkpoint=...)`.
9. Rank 0 only: save model/state/tokenizer, run `evaluate_slices` on all slices, write
   `eval_report.json` + `data_manifest.json`, push model + artifacts + generated README.

Key CLI flags: `--config`, `--smoke`, `--max-steps`, `--save-steps`, `--batch-size`,
`--grad-accum`, `--epochs`, `--backbone`, `--output-dir`, `--hub-model-id`,
`--push-to-hub`, `--seed`, `--resume`.

Strategy semantics:

- `--smoke` or `--max-steps > 0` uses step strategies. The step is
  `--save-steps` if given, else 15 when `max_steps <= 100`, else `max(50, max_steps//10)`.
- `--save-steps N` without `--max-steps` sets step save/eval with `num_train_epochs` from config.
- `--resume` means *continue*. It never silently retrains from scratch: no local checkpoint and
  no Hub checkpoint raises `SystemExit`.
- `--max-steps` with `--resume` counts *additional* steps from the checkpoint's `global_step`.
- Each resumed session re-anneals the cosine schedule over its own horizon (cyclic LR).

Resume (`_resolve_resume`): local `checkpoint-*` first; otherwise list Hub files, consider
`last-checkpoint/` and `checkpoint-<step>/`, rank by `trainer_state.json.global_step`, download
with `snapshot_download`, and require both state and weights. `hub.checkpoint_strategy:
checkpoint` means every save replaces `last-checkpoint/` (resumable across sessions); other
strategies are `end`, `every_save`, `all_checkpoints`.

Final evaluation uses `evaluate.py:evaluate_slices`; empty slices report `{"overall": {"n": 0}}`
and log a warning (they do not crash). Single-class slices produce NaN AUROC.

---

## 5. Data pipeline (`gradienttrace/data.py`)

### 5.1 Buckets and mix

`BUCKETS = ("human", "ai_foundational", "ai_frontier")`. Default mix is **0.40 / 0.40 / 0.20**.
`_targets` normalizes the mix and rejects unknown bucket names.

- `human`: label 0; generator forced to `"human"`.
- `ai_foundational`: label 1, generator not matching `frontier_patterns`.
- `ai_frontier`: label 1, generator matching `frontier_patterns` (`gpt-4`, `gpt-5`, `claude`,
  `gemini`, `o1/o3`, `llama-3`, `deepseek`, `qwen`, … — see config).
- `held_out_generators` (default `llama-2, mistral, gemini`) are never trained on; they are
  routed to `test_ood_generators`.

### 5.2 Sources

| spec | loader | notes |
|---|---|---|
| `hc3` | streaming `Hello-SimpleAI/HC3` `all.jsonl` | question + up to 2 human + 2 ChatGPT answers; prompt-paired; domain from `source` |
| `mage` | `yaful/MAGE` train split, loaded then shuffled | raw `label` is 1 = human, 0 = machine (the opposite of this pipeline); `src` is `<domain>_human` or `<domain>_machine_<prompt>_<model>`; yelp excluded because it is the gate's reviews source |
| `pile` | `artem9k/ai-text-detection-pile` | `source` column only distinguishes human vs ai; no generator, no prompt. v1 only — **not in v2/pilot** |
| `arena` | `lmarena-ai/arena-human-preference-55k` | `model_a/b`, `response_a/b` (JSON-list strings per turn); only public generator-labelled frontier source; domain `chat` |
| `hf:<org/name>[:split]` | generic adapter | auto-detects text/label/generator/prompt/domain columns; it assumes `label` 1 = AI, so check the dataset card before using it |
| `raid` | `cfg.data.raid_jsonl` | paired human/model rows, `variant` raw/humanized, `attack`, `group`; see 5.5 |
| `local_jsonl` | `cfg.data.local_jsonl` | user-supplied paired data; v2/pilot use the clean-news pairs from `colab/build_news.py`; see schema below |

Loader exceptions are caught per source, logged, and recorded in `manifest["errors"]`; one
flaky Hub dataset does not kill a run. An **unknown source name** (not matching any loader)
raises immediately from `_make_loader`, before that per-source try block.

`build_datasets` raises if it ends with no human samples or no AI samples; missing
`ai_frontier` only warns (but modern-model robustness depends on it).

### 5.3 Per-source quotas (`source_shares`)

`data.source_shares` gives each source a relative weight per bucket, e.g. v2:

| source | human | ai_foundational | ai_frontier |
|---|---|---|---|
| hc3 | 0.25 | 0.10 | — |
| mage | 0.35 | 0.30 | — |
| raid | 0.40 | 0.45 | 0.80 |
| arena | — | — | 0.40 |
| local_jsonl | 0.50 | 0.25 | — |

Behavior (`_select_by_share`):

- Weights are normalized across sources that actually have samples in that bucket.
- Quota = `min(pool_size, round(target * weight / total_weight))`.
- Any shortfall is redistributed to sources with spare capacity (up to 10 rounds).
- While loading, each source pool is capped at `int(target*share*1.5) + 64` to leave headroom.
- A source with no `source_shares` entry only fills shortfalls (warning logged).
- If no weights are usable, sources split the bucket evenly.
- `local_jsonl` is required in v2/pilot: if a weighted bucket loads zero rows, the build raises
  (missing `data/local.jsonl` would otherwise silently train the old confounded mix).

Consequence: raising a weight does not force data into the bucket; it only changes the quota
when that source has supply. Check `manifest["sources"]` and `manifest["train_sources"]`
after a build rather than assuming the mix landed.

### 5.4 Splits

- Dedup by normalized lowercase text hash across all sources; later duplicates dropped.
- `group_key(sample)` hashes `group` (RAID `source_id`) if present, else `prompt|domain`,
  else the text hash, into `[0, 10000)`.
- `val` if key < `val_frac*10000` (default 1000), `test_indomain` if < `(val+test)*10000`
  (default 2000), else `train`. Same prompt/group never straddles the split.
- `held_out_domains` (v2/pilot: `poetry`) are removed from training **before** selection; up to
  `ood_cap*2` of each class is kept for `test_heldout_domain`; the slice exists only if both
  pools are non-empty.

### 5.5 Slices and what each one means

| slice | composition | what it measures |
|---|---|---|
| `val` | 10% grouped split | threshold calibration + model selection (`load_best_model_at_end` on AUROC) |
| `test_indomain` | 10% grouped split | seen generators, unseen prompts — the easy number |
| `test_ood_generators` | held-out-generator AI + equal count of eval humans | unseen model families — the number that matters |
| `test_humanized` | humanized AI (reserve + test) capped at `ood_cap` + equal eval humans | paraphrased machine text |
| `test_heldout_domain` | paired AI/human from the held-out domain, `min(humans, ai, ood_cap)` pairs | domain generalisation |

Humans for OOD/humanized slices are drawn from val+test only, never from train.

`build_smoke_bundle` is synthetic and network-free; it has no `test_heldout_domain` key. Use it
only to verify the pipeline (`--smoke`), never to judge quality.

### 5.6 `data/local.jsonl` schema

```json
{"text": "human answer to prompt P", "label": 0, "generator": "human", "prompt": "P", "domain": "essay"}
{"text": "model answer to prompt P", "label": 1, "generator": "gpt-5", "prompt": "P", "domain": "essay"}
{"text": "paraphrased model answer to P", "label": 1, "generator": "gpt-5", "prompt": "P", "variant": "humanized"}
```

Optional keys the loader accepts: `domain`, `variant`, `bucket`, `group`. Pair AI with human on
the **same prompt**; unpaired corpora inflate in-domain scores and collapse under shift.

RAID JSONL records additionally carry `attack`, `pair_id` (equal to `group`), `source: "raid"`.
The loader ignores extra keys, so adding fields to the RAID builder is safe.

`colab/build_news.py` writes the same schema: `domain: news`, `bucket: ai_foundational` on the
AI rows, `source` = the dataset id (human) or `generated` (AI), and a `group: cnn-<hash>` shared
by a human lead and its AI texts so they never straddle a split. Its cc_news rows are
human-only (`label: 0`, `bucket: human`, `source: vblagoje/cc_news`, `group: cc-<hash>`, no AI
partner); the loader honours `bucket`, so they land in the human bucket regardless of the
missing pair. Rows can also carry `variant: "short"`: a 100–240-char first sentence of an AI
news text that mirrors the external gate's Qwen output shape. Short rows are AI-only: the
2026-11-01 pilots showed that short human news displaces the gate's long headline+wire form and
costs calibration, so `build_news.py` drops any legacy short human rows. The loader only
special-cases `humanized`; any other `variant` value passes through untouched, so short AI rows
are ordinary samples. They exist because `data.min_chars` is 100 in v2/pilot — the default 200
would drop every one.

### 5.7 Optional length matching

`data.length_match: true` rebalances the chosen pools so human and AI share a length histogram
over `length_bins` (`max_chars` appended as the last edge). It drops samples to do so and
records `manifest["length_match"]` with before/after counts. v2/pilot leave it off; the external
gate measures length effects separately.

---

## 6. Configs

All configs share the model/schedule; only data and Hub targets differ.

| | base (v1) | v2 | pilot |
|---|---|---|---|
| samples / epochs | 150k / 3 | 150k / 3 | 50k / 1 |
| sources | hc3, pile, mage, arena, local_jsonl | hc3, mage, arena, raid, local_jsonl | same as v2 |
| shares | none | per 5.3 (local human 0.50, AI 0.25) | per 5.3 (local human 0.50, AI 0.25) |
| min chars | 200 | 100 | 100 |
| RAID | no | `data/raid_v2.jsonl` | same |
| news pairs | — | `data/local.jsonl` from `build_news.py` | same |
| held-out domain | — | poetry | poetry |
| output dir | `runs/gradientTrace` | `runs/gradientTrace-v2` | `runs/gradientTrace-v2-pilot` |
| Hub repo | `Yashhh999/gradientTrace` | `…-v2` | `…-v2-pilot` |
| early stopping | patience 0 | 0 | 0 |
| length match | — | false | false |

`configs/v2_ailow10.yaml` is v2 with `local_jsonl.ai_foundational` 0.25 -> 0.10 only, run at the
pilot's 607-step budget to test whether per-batch exposure to the journalist-prompt/short-AI
contrast causes the 150k human-calibration regression. It does partly, and the config is
archived as that experiment's record (see 7.5).

`early_stopping_patience: 0` is deliberate: with step-based evaluation, every eval is a
patience tick, and v1 early-stopped at step 5,607 (~1 epoch) when val AUROC saturated. Keeping
patience 0 forces the configured number of epochs.

`hub.private: true` for the v2 repos. `hub.checkpoint_strategy: checkpoint` for all.

---

## 7. Execution surfaces

### 7.1 Local

```bash
pip install -r requirements.txt
python -m gradienttrace.train --config configs/base.yaml --smoke --max-steps 40
python -m gradienttrace.train --config configs/base.yaml [--push-to-hub] [--resume]
python -m gradienttrace.evaluate --model runs/gradientTrace --config configs/base.yaml
python -m compileall -q gradienttrace colab tests && python tests/test_metrics.py
```

Do not execute `gradienttrace/*.py` as a script in a notebook; the package uses relative
imports. `ImportError: attempted relative import with no known parent package` means the file
was pasted/run outside package context. Use `python -m …` or `from gradienttrace.data import …`
after `sys.path.insert(0, repo_dir)`.

### 7.2 Kaggle cell (`colab/one_shot_kaggle.py`)

Settings: **Accelerator = GPU T4 x2**, **Internet = On**, `HF_TOKEN` in Add-ons > Secrets.

- Knobs: `RUN` (`smoke|full`), `CONFIG` (`base|v2|pilot`), `HUB_ID` (empty = per-config default),
  `MAX_STEPS`, `SAVE_EVERY`, `EPOCHS`, `RESUME`, `MICRO_BATCH=4`, `GRAD_ACCUM=8`,
  `LOCAL_JSONL`.
- Launches `torch.distributed.run --standalone --nproc_per_node 2` (4×8×2 = 64 global batch,
  fp16, gradient checkpointing).
- For `v2|pilot` it downloads `v2/raid_v2.jsonl` from `RAID_HUB` and **refuses a slice with
  fewer than 10,000 humans or fewer than seven populated domains** (the first stale slice had
  1,233 humans in three domains and would otherwise train silently). It then downloads
  `v2/news_pairs.jsonl` from `NEWS_HUB` into `data/local.jsonl` and refuses fewer than 1,000
  human or 1,000 AI rows (unless `LOCAL_JSONL` overrides it). The news pairs are the clean
  human-news contrast; training without them re-learns "clean = AI".
- With `push`, it runs a **Hub preflight** before training: prints token role, creates the repo
  if needed, uploads and deletes `.preflight`. A bad token fails in seconds, not after a 2-hour
  run.
- `run()` streams child output and, on failure, reprints the last traceback at the bottom of the
  cell output (torchrun prints it above the `Failures:` banner, where it is easy to lose).
- Sessions cap at 12h and `/kaggle/working` is wiped between sessions: resume is Hub-only.
  To restart inside a live session, kill leftovers (`pkill -f torch.distributed.run`,
  `pkill -f kaggle_launch.py`) and delete `runs/`, or the old torchrun keeps the GPUs.

Measured: ~7.7 s/step at global 64; val eval over ~8.6k samples ~4 min; save + Hub push ~40 s.

### 7.3 Colab cell (`colab/one_shot.py`)

GPU or TPU v5e-1. Hardcodes `configs/base.yaml`; it has **no CONFIG knob** — use the Kaggle cell
for v2/pilot. TPU constraints, all encoded in the cell and `runtime.py`:

- The image uses the native `torch_tpu` backend (compiled extensions only for torch 2.11–2.15).
  A stale `torch_xla` hijacks detection and its `_XLAC.so` is ABI-locked; the cell uninstalls it.
- Installing `torch_xla` downgrades `libtpu` to a PJRT plugin too old for torch_tpu; the cell
  restores the libtpu `torch_tpu` declares (else newest).
- **Never `import torch` in the TPU kernel.** It claims the chip and the training child can no
  longer initialize libtpu (silent exit 1). Verify in a short-lived child instead, including
  `torch.manual_seed(0)` and a real `tpu:0` op, which is where a bad libtpu surfaces.
- `train.py` sets `ACCELERATE_TORCH_DEVICE=tpu:0`; runtime kind `tpu`, device `tpu:0`, bf16,
  micro 4 × accum 16, static padding, 0 workers.

Measured: ~16 s/optimizer step at effective batch 64; a 150k×3 run is ~30h, spanning sessions.

### 7.4 Finish eval (`colab/finish_eval.py`)

Evaluates the Hub model slice by slice, saves `runs/gradientTrace/eval_partial.json`, and pushes
it to the Hub after each slice, so a disconnect costs at most the slice in flight. fp16,
length-sorted batches. Knobs: `HUB_ID`, `BATCH_SIZE`, `PUSH`, `RESUME_PARTIAL`.

It hardcodes `configs/base.yaml`, so for a v2 model change that line to `configs/v2.yaml`
**and** provide `data/raid_v2.jsonl`, otherwise the eval slices are the v1 mix. Training already
evaluates the v2 slices at the end of a run; use this cell mainly to finish a killed run.

### 7.5 External gate (`colab/external_gate.py`)

The acceptance test. It builds a **frozen** matched dataset and scores any detector on it:

- Humans: 400 `fancyzhx/ag_news` test (news) + 400 `Yelp/yelp_review_full` test (reviews).
- AI: `Qwen/Qwen2.5-3B-Instruct` (unseen in training; RAID's qwen-1.5 rows are held out)
  writes one response per human sample, using the human's snippet as the topic and matching
  its length; sampling T=0.8, top_p=0.9, batch 8.
- Frozen as `external_gate_v1.jsonl` on `Yashhh999/gradientTrace` and restored on later runs, so
  every model is scored on identical text.
- Metrics per genre and pooled: AUROC, TPR at a human-calibrated 1% FPR (headline), TPR/FPR at
  0.5, pairwise accuracy, length-matched AUROC, length buckets.

To gate a new model: set `MODEL_ID` and `MODEL_LABEL`, keep `GATE_NAME="external_gate_v1"`,
`REGENERATE=False`, `PUSH=True`. Do not regenerate the dataset between model comparisons.
`PUSH=False` on a fresh VM would regenerate all AI text — expensive and not comparable.

Recorded baseline (`external_gate_v1_report_v1.json`, 2026-10-31):

| slice | n per class | AUROC | TPR@1% FPR | FPR@0.5 | pairwise | matched AUROC |
|---|---|---|---|---|---|---|
| news | 400 | 0.4587 | 0.0050 | 0.9550 | 0.4125 | 0.5424 |
| reviews | 400 | 0.6387 | 0.0000 | 0.9950 | 0.4500 | 0.6532 |
| pooled | 800 | 0.5270 | 0.0000 | 0.9750 | 0.4313 | 0.5428 |

Interpretation: v1 is unusable outside its training sources. TPR@1% FPR is the number to quote.

First v2 pilot (`external_gate_v1_report_v2-pilot.json`, 2026-10-31): pooled AUROC 0.5745,
TPR@1% FPR 0.0263, FPR@0.5 0.9675, pairwise 0.5950, matched AUROC 0.5903; news 0.5685,
reviews 0.6345. Directionally better than v1 but unusable (96.75% of human texts still score
above 0.5). The pilot trained with MAGE labels inverted (see §11), so this is not a clean
measurement of the recipe; the next pilot supersedes it.

v2 pilot with fixed MAGE labels (`external_gate_v1_report_v2-pilot-fix.json`, 2026-11-01):
pooled AUROC 0.7258, TPR@1% FPR 0.1288, FPR@0.5 0.5463, pairwise 0.7100, matched AUROC 0.6387;
news 0.4775 / 0.0125 / 0.9700 (human mean P(AI) 0.9208, AI mean 0.9318); reviews 0.9653 /
0.4525 / 0.1225. The pooled move is carried entirely by reviews — Yelp, which RAID supplies in
training. On news, the only slice whose humans come from a source never in training, the model
is at chance and still flags 97% of human text as AI. Internal held-out poetry jumped to
0.9732 / 0.577, but every poetry human row in the RAID slice is perturbed, so that is partly
the same shortcut (§7.6).

Clean-news pilot (`external_gate_v1_report_v2-pilot-news.json`, 2026-11-01): the first run
whose mix actually contains the corpus — local_jsonl 6,903 train rows (3,224 human + 3,679 AI;
SmolLM2 1,890 + Phi-3 1,789 in `train_generators`), human bucket 16,056, so 20% clean news.
Gate: news 0.5906 / 0.0350 / 0.9300 (human mean P(AI) 0.9180, AI mean 0.9653), reviews 0.9429 /
0.4000 / 0.1925, pooled 0.7319 / 0.0963 / 0.5613. The news AUROC gain is almost entirely the AI
side (0.9318 -> 0.9653), and part of that is prompt recognition: the training AI news was written
with the gate's own journalist prompt. The human mean barely moved (0.9208 -> 0.9180) and 93% of
clean ag_news humans still score above 0.5, so the gate is not cleared. The pilot's own
`domain:news` slice is 0.998 on val and test_indomain — in-domain news is solved; the gate's
ag_news form (headline + wire description) is unseen. Only 20% of the human bucket is clean news
and more CNN would repeat the same source; the next test adds human news from many outlets in
the gate's title+description form.

cc_news pilot (`external_gate_v1_report_v2-pilot-ccnews.json`, 2026-11-01): the human-form fix
works. The corpus gained 6,000 human-only cc_news title+description rows and `local_jsonl.human`
rose to 0.5, so local_jsonl supplied 9,038 train rows (human bucket 16,064, local ~5k = 31% of
it; the extra weight displaced hc3/mage/raid rows because the bucket target is 16k). Gate: news
0.8131 / 0.2325 / 0.3575 (human mean P(AI) 0.9180 -> 0.3746, p50 0.9942 -> 0.2457; AI mean
0.9653 -> 0.7876), reviews 0.9293 / 0.3725 / 0.1500, pooled 0.8725 / 0.2637 / 0.2537. News
AUROC +0.22, FPR@0.5 -0.57: the headline+wire form was the human-side gap. The weak side is now
the AI side — only 23% of Qwen news is caught at 1% FPR, 19% of AI scores below 0.5, and the
1% threshold sits at human p99 = 0.9958. Reviews barely moved (0.9429 -> 0.9293). Not shippable,
but this is the first model that separates news at all.

Short-variant pilot (`external_gate_v1_report_v2-pilot-short.json`, 2026-11-01): the length fix
moved the AI side. Corpus 14,000 human / 14,984 AI (7,001 short AI texts); local_jsonl supplied
10,512 train rows and news became 14,017 of 38,636 train rows (36%). Gate: news 0.9108 / 0.5400
/ 0.4975 (human mean 0.3746 -> 0.5095, p50 0.2457 -> 0.4824; AI mean 0.7876 -> 0.9359; the 1%
threshold is 0.9968 = human p99), reviews 0.9526 / 0.4950 / 0.1175 (improved), pooled 0.9032 /
0.4263 / 0.3075, pairwise 0.9463, matched 0.8761. News AUROC +0.10 and TPR@1% FPR +0.31 — 54%
of Qwen news is caught when clean humans are allowed 1% false positives — but FPR@0.5 rose 0.14:
the model now calls half the clean humans AI at the default threshold. Cause: short AI rows were
47% of the AI corpus while short humans were only 29% and every short human was an unpaired
cc_news brief, so the model learned "short news = AI" on top of the form fix.

The balanced-shorts rerun (`external_gate_v1_report_v2-pilot-short2.json`, 2026-11-01) refuted
that fix: 8,589 short human rows (6,000 cc briefs + 2,589 paired CNN first sentences) and 7,002
short AI made the human side worse. Gate: news 0.8994 / 0.4000 / 0.5925 (human mean 0.5095 ->
0.5926, p50 0.4824 -> 0.6754; AI mean 0.9359 -> 0.9422; 1% threshold 0.9983 = human p99),
reviews 0.9585 / 0.5200 / 0.1150, pooled 0.8937 / 0.3862 / 0.3538 (pair 0.9437, matched
0.8624). The 0–400 bucket human mean rose 0.514 -> 0.594 and 400–800 rose 0.418 -> 0.592, so
this is not short-text confusion. The local human quota is fixed (6,667 rows) and shuffled, so
each short row added displaced a long one: long local train fell ~3,800 -> ~2,900 while the
gate's humans are 247-char headline+wire. The long-only cc_news model (5,333 long local humans,
human mean 0.3746) calibrated the same 0–400 humans best, and the short pilot (0.9108 AUROC,
0.5400 TPR@1%) remains the best news model. Next: short AI + long-only human news.

Longhuman pilot (`external_gate_v1_report_v2-pilot-longhuman.json`, 2026-11-02): the best news
model so far. The corpus is 10,000 long humans (4,000 CNN leads + 6,000 cc wires) and 15,002 AI
texts (7,002 short); the pilot's 6,667-row local human quota is 100% long (~5,333 train). Gate:
news 0.9323 / 0.6175 / 0.4225 (human mean 0.5095 -> 0.4273, p50 0.2526, p90 0.9848, p99 0.9978;
AI mean 0.9406; 1% threshold 0.9978 = human p99), reviews 0.9151 / 0.3125 / 0.2275 (human mean
0.2425, AI mean 0.8608), pooled 0.9088 / 0.4412 / 0.3250 (pair 0.9525, matched 0.8786). The
news AUROC and TPR@1% are the best recorded and TPR@1% clears 0.5 for the first time, but
FPR@0.5 is 0.06 above the 0.36 bar. Removing the short human rows recovered only ~0.08 of the
0.22 that short2 cost (0.5926 -> 0.4273): the news residual is concentrated in 0–400 chars
(human mean 0.4323, n=386) while 400–800 is 0.3103 (n=13), so the short AI rows still teach
"short = AI". Reviews lost the short-clean-human calibration (TPR@1% 0.4950/0.5200 -> 0.3125),
the cost of an all-long local human mix.

Longhuman2 pilot (`external_gate_v1_report_v2-pilot-longhuman2.json`, 2026-11-02): raising
`local_jsonl.human` to 1.0 (local human quota 10,000, ~8k train) refuted the "more clean humans"
hypothesis and regressed the news bar. Gate: news 0.9299 / 0.4825 / 0.5325 (human mean 0.4273 ->
0.5431, p50 0.2526 -> 0.6039, p90 0.9957, p99 0.9987; AI mean 0.9406 -> 0.9699; 1% threshold
0.9987), reviews 0.9697 / 0.5450 / 0.1700 (human mean 0.1949, AI 0.9377 — a large gain, but Yelp
is seen via RAID), pooled 0.9278 / 0.4412 / 0.3513 (pair 0.9725, matched 0.8780). AUROC is flat
while both means rose, so this is a calibration shift: the larger local news-AI contrast
sharpened AI scores (0.94 -> 0.97) and spilled onto clean ag_news humans; the news human
distribution moved up as a whole (0–400 0.432 -> 0.546, 400–800 0.310 -> 0.495), not just at
short lengths. Internal val 0.9964 / 0.927, OOD 0.9991, humanized 0.9999, poetry 0.9532 / 0.503.
The 0.5 human mix remains the best news model; the residual is the strength of the local
journalist-prompt/short-AI contrast, not a shortage of clean humans.

Ailow pilot (`external_gate_v1_report_v2-pilot-ailow.json`, 2026-11-02): cutting local AI 0.4 ->
0.25 cleared the bar and is the best news model. Human weight back at 0.5 and the corpus is
unchanged, so the human data is identical to longhuman. Gate: news 0.9487 / 0.7925 / 0.2475
(human mean 0.2623, p50 0.0558, p90 0.9193, p99 0.9922; AI mean 0.9196; 1% threshold 0.9928),
reviews 0.9360 / 0.3650 / 0.1375 (human mean 0.1639, AI 0.8004 — Yelp-seen), pooled 0.9323 /
0.5887 / 0.1925 (pair 0.9575, matched 0.9053; human 0.2131, AI 0.8600). The clean-human score
fell 0.4273 -> 0.2623 with no human-data change, in both length buckets (0–400 0.432 -> 0.263,
400–800 0.310 -> 0.268): the residual cue was the strength of the local journalist-prompt AI
contrast, as longhuman2 predicted. TPR@1% +0.175 with AUROC +0.016, so the gain is tail
calibration, not ranking. Reviews TPR@1% 0.365 is below the 0.40 watch line but source-seen.
Internal val 0.9961 / 0.948, in-domain 0.9959 / 0.909, OOD 0.9990 / 0.982, humanized 0.9998 /
0.991, poetry 0.9624 / 0.538. **Clears the pilot bar.** The 150k needs the local human corpus at
its 20,000-row quota (16,000 cc + 4,000 CNN) or the local share halves to 1/6 of the human bucket.

150k gates (`..._report_v2-full.json`, `_v2-1ep.json`, `_v2-150k607.json`, `_v2-ailow10.json`,
2026-11-02/03): **none clears the bar, and the failure is calibration, not ranking.** News
AUROC stays 0.93-0.95 while the human mean sits at 0.62-0.68 (FPR@0.5 0.60-0.71) for the three
v2-recipe builds at 5,265 / 1,755 / 607 steps — the step budget is not the variable. The fourth
build cut `local_jsonl.ai_foundational` to 0.10 at the same 607 steps: news 0.9176 / 0.2825 /
0.2825 (human mean 0.616 -> 0.362, AI mean 0.9639 -> 0.9082), reviews 0.9560 / 0.5450 / 0.3300,
pooled 0.9371 / 0.4612 / 0.3063. So the local news-AI share *is* the knob that spills onto clean
humans (FPR@0.5 0.600 -> 0.2825, under the 0.36 bar) and the human side recovers, but the AI tail
loses separation with it (TPR@1% 0.665 -> 0.2825 against a 0.60 bar). Data volume, not steps,
moved the whole 150k human distribution up; the 50k pilot remains the release model and the 150k
data-mix search is closed. Internal slices for ailow10: val 0.9948 / 0.907, in-domain 0.9949,
OOD 0.9993 / 0.990, humanized 0.9975 / 0.951, poetry 0.9507 / 0.435.

Gate caveat: the reviews humans are Yelp and RAID's reviews humans are also Yelp, so the
reviews slice is source-seen for v2 models; news (ag_news vs RAID's BBC News) is the cleaner
human-source slice. The generator Qwen2.5-3B-Instruct is unseen after `qwen` is held out.

The cell aborts when `MODEL_ID` is still `GATE_REPO` while `MODEL_LABEL` is not `v1`: on
2026-11-01 a `v2-pilot-short2` run left `MODEL_ID` at the default and silently re-scored v1
(its report reproduced the v1 baseline exactly, with `"model": "Yashhh999/gradientTrace"`).

### 7.6 RAID extractor (`colab/build_raid.py`)

One-time, resumable, CPU-only. RAID `train.csv` is an 11.8 GB CSV with domains in contiguous
blocks; the cell reads only the byte ranges in `WINDOWS` via HTTP Range (206 only), keeps rows
for the requested domain, and pushes `v2/raid_v2.jsonl` + `v2/raid_v2_meta.json`.

- Caps per domain: 3,000 humans; 6,000 raw AI; 900 per model; 1,500 humanized; 300 humanized per
  model; 1,500 per human attack.
- Human rows keep RAID's surface perturbations (spelling/whitespace/…) because RAID still labels
  them human; `paraphrase`/`synonym` humans are excluded (they would blur with humanized AI).
- `MIN_CHARS=200`, `MAX_CHARS=12000`; whitespace normalized.
- A domain with 0 matching rows is **not** marked complete (wrong byte window guard).
- `REGENERATE=True` wipes local files and rebuilds every domain; **the committed value is still
  True after the successful rebuild — set it to False before rerunning.**
- Byte windows were probed: books 1.75, news 3.30, poetry 5.05, recipes 6.20, reddit 7.85,
  reviews 9.04, wiki 9.93 GB. Earlier windows silently read the wrong domains.

Current slice (checked on the Hub 2026-10-31): 71,297 rows, 18,848 human rows across 7 domains;
133.2 MB. The 73,500 / 21,000 figures printed during the build are caps, not actual rows. Human
p50 885–1,922 chars; AI p50 892–1,962. Clean (`attack: none`) humans exist only in reviews
(650), wiki (293), recipes (290). News, reddit, books and poetry are 100% perturbed humans
(news: 1,500 `perplexity_misspelling` + 1,500 `number`, zero clean) while every AI row is
clean — so in those domains perturbation is the only cue separating the classes, and the model
learns "clean = AI". This is the gate's news failure (§7.5), not a modelling problem.

### 7.7 Clean-news builder (`colab/build_news.py`)

One-time, resumable, GPU cell. It fixes the news failure from the human side: CNN/DailyMail
leads (clean, unperturbed human news prose, 200–520 chars, cut on a sentence boundary) are each
paired with one AI news text per generator, written with the external gate's own journalist
prompt (same system line, same `{words}`/snippet instruction, T=0.8, top_p=0.9, length-matched
and trimmed at a sentence boundary).

- Humans: `nschantz21/cnn_dailymail-parsed` (parquet copy of CNN/DailyMail 3.0.0), train split,
  `article` field; first sentences until 200 chars, max 520. A lead that does not end on a
  sentence boundary is skipped.
- Extra humans (human-only, no generation): `vblagoje/cc_news` (`plain_text` config, train
  split) `title` + `description`, one sentence of body appended if the pair is still under 200
  chars, cut on a sentence boundary at 520. This is the gate's ag_news form — a headline plus a
  wire description — from many outlets, and it is why `local_jsonl.human` is 0.5 (1.0 on
  2026-11-02 regressed news calibration: TPR@1% 0.6175 -> 0.4825, FPR@0.5 0.4225 -> 0.5325).
  Rejected: titles over 520 chars, texts with no sentence-ending punctuation, and duplicates
  of anything already in the file (dedup by normalized lowercase text).
- Short AI variants close the length/form gap that made the cc_news pilot's AI side fail. The
  gate's Qwen news is 1.3 sentences (p5/p50/p95 = 78/134/223 chars) while every training AI row
  was 220+ chars (p50 331, 3.8 sentences) — zero overlap, so short clean text read as human.
  `add_short_variants()` writes the first sentence of each AI text (100–240 chars, i.e. the
  gate's shape) as a `variant: short` row. Human news stays long: the 2026-11-01 builds showed
  the gate's clean-human mean rising 0.375 -> 0.510 -> 0.593 as short human rows displaced long
  headline+wire humans, so the function drops every `variant: short` human row and never writes
  one. The AI section is rewritten interleaving short and long rows, because the per-source pool
  cap in `data.py` takes rows in file order and would otherwise see only one variant.
- Generators: `microsoft/Phi-3-mini-4k-instruct` and `HuggingFaceTB/SmolLM2-1.7B-Instruct`.
  Granite-2B was dropped: its fp16 logits went NaN on the T4 (`probability tensor contains
  either inf, nan or element < 0`). Never Qwen — it is the gate's generator.
- Generation forces `min_new_tokens ≈ the human's token count` because Phi-3-mini often stops
  after ~25 words, which put most outputs under the 200-char loader minimum (first pass kept
  only 14%). Every progress line prints `[news] rejects: {...}` with the drop reasons and a
  couple of examples, so a low yield is diagnosable instead of silent. A model's rows are
  pushed when it finishes, even if the next model crashes the CUDA context.
- Output: 4,000 CNN human leads + 16,000 long cc_news human wires (6,000 until the 150k
  top-up) + up to 8,000 long AI texts (one per story per generator) + one short AI variant per
  text as `v2/news_pairs.jsonl` (20,000 human / ~15,000 AI once topped up; 10,000 / 15,002
  after the 2026-11-02 long-human build), plus
  `v2/news_pairs_meta.json` (counts, `human_sources`, per-generator lengths including the short
  variants, prompt template). Rows use the `local.jsonl` schema: label 0/1, `domain: news`,
  `bucket: ai_foundational` for AI and `human` for humans, and a shared `group` per story so
  the human and both AI rows never straddle a split. Short AI rows copy their parent's group.
- Resumable: rows are appended and pushed every 200 generations; `REGENERATE=False` restores
  the pushed file, and a rerun tops up missing stories per model and missing cc_news rows (it
  counts existing `source: vblagoje/cc_news` rows first, so a rerun does not duplicate them).
  The short stages are idempotent: an existing short cc row or AI variant is never re-added, and
  a no-op rerun leaves the file byte-identical (verified on a 100-row sample).
- Wired in by `one_shot_kaggle.py`: for `v2`/`pilot` it downloads the pairs to
  `data/local.jsonl` and aborts if they are missing (unless `LOCAL_JSONL` is set). Its guard is
  1,000 human and 1,000 AI rows, so the file must contain the AI side too — cc_news alone is not
  enough. `configs/{v2,pilot}.yaml` give `local_jsonl` 0.50 of the human bucket and 0.25 of
  AI-foundational (a 6,667-row human quota at the pilot, ~5.3k train; 20,000 at the 150k,
  ~16k train) with `min_chars: 100` so the short variants survive the loader. As a second line
  of defence, `build_datasets` raises if a weighted `local_jsonl` bucket loaded zero rows, so a
  stale notebook cell cannot silently train the old mix.
- Cost: ~30–60 min per generator on one T4 for 4,000 stories (the cc_news stage is ~10–20 min
  of dataset streaming, no GPU). The 2026-11-02 prune was a restore-and-rewrite (~1–2 min); the
  150k top-up replays the cc_news stream for 10,000 more rows (~20–30 min, no GPU) and appends
  them idempotently. The dataset is then frozen and reused by later runs. **A rebuilt news
  file is a data change: set `RESUME=False`.**

### 7.8 Hybrid eval (`gradienttrace/hybrid.py`, `colab/hybrid_eval.py`)

Second opinion computed from the detector itself — there is no second model. Feature blocks, in
`FEATURES` order:

- `det_logit` = `log(p / (1 - p))`.
- `stylometry` — 33 deterministic surface statistics: function-word / pronoun / determiner /
  preposition / conjunction / auxiliary / connective / LLM-marker rates (per 100 words), digit,
  uppercase, all-caps, contraction rates, one rate per punctuation class (comma, period, colon+
  semicolon, quote, parens, dash, ellipsis, exclaim, question), markdown and newline rates,
  mean and **std of sentence lengths** (burstiness), mean word length, long-word rate,
  syllables/word, Flesch, TTR, hapax rate, repeated-trigram fraction.
- `vector_features` — geometry of the detector's own pooled embedding (k=8): mean cosine to the
  nearest k labelled humans, to the nearest k AI rows, their margin, the human fraction among the
  k nearest overall, and the centroid-distance margin. The scalar logit throws this space away.

Four rules are what make the numbers mean anything:

- **Vector features are rebuilt inside every CV fold** from training-fold references only
  (`oof_fusion`). A feature matrix computed once over the whole eval set would leak labels.
- **References sharing a group are blocked.** The gate is prompt-paired, so a row's nearest
  neighbour is its own pair partner — that measures topic, not style. `tests/test_hybrid.py`
  asserts both rules; the assertion fails loudly if blocking is removed.
- `predict_probs_and_embeddings` recomputes the logit from the pooled vector instead of calling
  `forward`, which is bit-identical in eval mode and halves the cost. Asserted against
  `predict_probs` in `tests/test_hybrid_inference.py`.
- Report the share of folds agreeing on each coefficient's **sign**. The fusion gives `det_logit`
  a negative weight with 5/5 fold agreement: the geometry carries the signal and the logit
  survives as a corrector. Stable, but do not read it as "the detector is inverted".

Knobs: `DET_ID` (Hub id **or** a local directory), `GATE_REPO`/`GATE_NAME`, `PROBE_TXT`
(explainer blocks `**N. Title`; `""` skips the transfer stage), `OUT_DIR`, `K_NEIGHBOURS`.
Runs on CPU: 1,600 gate rows in ~2.5 min at 1.4-2.0 s per 16-row batch, so no GPU session is
needed for any of §7.8.

Recorded (ailow pilot as detector, 5-fold out-of-fold on the frozen gate, 2026-11-03; the
detector row reproduces `external_gate_v1_report_v2-pilot-ailow.json` exactly, which validates the
harness; every coefficient kept its sign in 5/5 folds):

| feature set | pooled AUROC | TPR@1% | TPR@5% | FPR@0.5 | news AUROC | news TPR@1% | news FPR@0.5 | reviews TPR@1% |
|---|---|---|---|---|---|---|---|---|
| detector only | 0.9323 | 0.5887 | 0.6925 | 0.1925 | 0.9487 | 0.7925 | 0.2475 | 0.3650 |
| stylometry only | 0.9378 | 0.3438 | 0.6075 | 0.1525 | 0.9503 | 0.4075 | 0.1200 | 0.2475 |
| vectors only | 0.9843 | 0.7063 | 0.9263 | 0.0737 | 0.9938 | 0.9200 | 0.0125 | 0.4100 |
| det + stylometry | 0.9699 | 0.5100 | 0.8575 | 0.0912 | 0.9699 | 0.6800 | 0.0875 | 0.5325 |
| det + vectors | 0.9850 | 0.7175 | 0.9313 | 0.0775 | 0.9942 | 0.9300 | 0.0125 | 0.4175 |
| hybrid all | **0.9876** | **0.7688** | **0.9475** | **0.0550** | 0.9910 | 0.9200 | 0.0275 | **0.6000** |

Cross-register transfer, each system at **its own** FPR-calibrated operating point (detector
0.9872, hybrid 0.9464 — the fused score is not calibrated to 0.5, so never compare at a shared
threshold):

- 20 frontier explainer texts: detector flags 1/20, hybrid 9/20 (at 0.5: 13/20 vs 17/20).
- 14 mixed texts (6 AI, 8 human — Gettysburg, Alice, Moby-Dick, the Declaration, Conan Doyle,
  Darwin, two Wikipedia leads) at 0.5: detector AI 4/6 + humans 4/8 flagged; hybrid AI 4/6 +
  **humans 2/8**.

Readings (all of them superseded where they contradict the external-bank table below):

- The embedding geometry dominates the gate-fitted fusion; stylometry alone is barely better than
  the detector (0.9378) and contributes sentence-shape and punctuation cues (`sent_mean_words
  -1.67`, `hapax_rate +1.17`, `ellipsis_rate -0.96`).
- Superseded: the GPT-2 reference-LM fusion (pooled AUROC 0.9481, 2/20 explainers). Dropped
  because it calibrated without extending coverage and its `bpb` feature ranked archaic and
  encyclopedic prose as machine text (Declaration of Independence bpb 0.163 vs AI explainers
  0.89-1.27).

### 7.9 The external-bank re-test (`colab/hybrid_bank_eval.py`)

The table above fits and scores on the frozen gate, so its k-NN bank **is** the population being
measured. Out-of-fold CV stops a row from seeing its own label and blocks pair partners, but the
human/AI clusters are still the test set's own clusters. `hybrid_bank_eval.py` rebuilds the bank
from the pilot's held-out slices (`test_indomain` + `test_ood_generators` + `test_humanized` +
`test_heldout_domain`, 4,575 rows balanced by slice/label/domain), fits the fusion there, sets the
threshold from the **bank's** out-of-fold scores, and then scores the gate as a population it has
never seen.

| feature set, gate as unseen population | pooled AUROC | pooled TPR@1% | pooled FPR@0.5 | news AUROC | news TPR@1% | news FPR@0.5 | reviews AUROC | reviews TPR@1% | reviews FPR@0.5 |
|---|---|---|---|---|---|---|---|---|---|
| detector | 0.9323 | 0.5887 | 0.1925 | 0.9487 | 0.7925 | 0.2475 | 0.9360 | 0.3650 | 0.1375 |
| det + stylometry | **0.9422** | 0.5787 | **0.0638** | 0.9480 | **0.8050** | **0.0825** | **0.9494** | **0.3700** | **0.0450** |
| det + vectors | 0.9105 | 0.5038 | 0.0775 | 0.9502 | 0.8125 | 0.1125 | 0.9060 | 0.2125 | 0.0425 |
| hybrid all | 0.9136 | 0.5125 | 0.0950 | 0.9488 | 0.8050 | 0.1250 | 0.9091 | 0.2400 | 0.0650 |

- **Stylometry transfers; embedding geometry does not.** `det + stylometry` beats the detector on
  nearly every axis and halves-to-two-thirds the false accusations at 0.5 in both genres. Every
  set containing k-NN features falls *below* the bare detector on pooled AUROC (0.9105 / 0.9136 vs
  0.9323) and collapses reviews TPR@1% (0.3650 -> 0.2125). The +0.074 AUROC the geometry produced
  in-bank was the fusion recognising the evaluation set's own clusters.
- **The coverage claim was cluster memory too.** Under the external bank the fusion flags 0-1 of
  the 20 frontier explainers at its bank-calibrated cut, the same as the detector's 1/20 — not the
  9/20 measured in-bank.
- **Shipped set: `det + stylometry`** (`SHIP_SET` in `colab/hybrid_eval.py`). It needs no
  `bank.npz` at inference, so deployment is the detector plus a pure feature function;
  `HybridScorer` notices the fusion has no vector columns and skips the bank entirely (asserted in
  `tests/test_hybrid_inference.py`). At the bank-calibrated 0.8741 cut it catches 459/800 gate AI
  texts with 8 human false alarms (exactly 1.0% FPR) versus the detector's 471/800: parity on
  detection, two thirds fewer accusations, on a population it never saw.
- Encyclopedic register still fails regardless of scorer: Wikipedia's Great Barrier Reef lead
  scores 0.9695 as AI. That is a data gap (§11), and it is the reason the next real investment is
  paired register coverage, not another fusion.

Both runs cache what they load (`hybrid_bank_rows.json`, `hybrid_bank_cache.npz`,
`hybrid_gate_cache.npz` — all gitignored), so re-scoring the fusion costs seconds; the ~19 minute
data build is the only expensive step and it runs once.

Using it: `save_fusion(OUT_DIR, fusion, embeddings, labels, groups, DET_ID, k)` writes
`fusion.json` (feature order, standardizer, weights, threshold) plus `bank.npz` (embeddings,
labels, groups); `HybridScorer(detector_dir, fusion_dir).score(texts)` returns `fused`,
`detector`, `threshold` and `flagged`. Both files are artifacts — never commit them. For the
shipped `det + stylometry` set the fusion has no vector columns, so `HybridScorer` loads
`bank.npz` and then ignores it: the file is written but carries no weight at inference. Deleting
it next to a deployed fusion changes nothing; keeping the pair matching is what matters.

---

## 8. Metrics and how to report them

`metrics.py`:

- `binary_detection_metrics`: AUROC, AUPRC, accuracy@0.5, TPR@1%/5% FPR, best threshold (Youden),
  ECE. Single-class input yields NaN for AUROC/AUPRC/accuracy/threshold/ECE rather than raising.
- `sliced_report`: overall plus one entry per value of each metadata field
  (`generator:<name>`, `domain:<name>`).

Reporting rules:

- Never quote internal AUROC ≈ 0.999 as detection quality; it is source-confounded (see §1).
- Lead with TPR@1% FPR from the gate, then AUROC, pairwise, and matched AUROC.
- State TPR@1% FPR as “AI caught when humans are allowed a 1% false-positive rate”.
- A model is not “done” because internal slices look perfect; it is done when the gate moves.
- For any fusion (§7.8): vector features must be rebuilt inside each CV fold with pair partners
  blocked, coefficients reported with their fold sign-agreement, and each system compared at **its
  own** FPR-calibrated threshold — a fused score is not calibrated to 0.5.
- Say whether a fusion's reference bank was outside the evaluated rows. Fitting and scoring on the
  same population is an upper bound, not an acceptance result.
- When asked about a live run, answer verdict-first (healthy / not healthy), name only real
  problems, and mention known-harmless noise in one line. State decisions (stop/resume) first.

Known-harmless log noise: duplicate per-rank lines, NCCL `barrier()`/`destroy_process_group()`
warnings, tokenizer PAD/BOS/EOS alignment notes, LOAD REPORT `UNEXPECTED` MLM heads, brief
step-time spikes at checkpoint saves.

---

## 9. Hub artifacts

| repo | contents |
|---|---|
| `Yashhh999/gradientTrace` | v1 model at root + `checkpoint-5607/`, `last-checkpoint/`, `eval_report.json`, `data_manifest.json`, README, `external_gate_v1.jsonl`, `external_gate_v1_meta.json`, every gate report (`_v1`, `_v2-pilot`, `_v2-pilot-fix`, `_v2-pilot-news`, `_v2-pilot-ccnews`, `_v2-pilot-short`, `_v2-pilot-short2`, `_v2-pilot-longhuman`, `_v2-pilot-longhuman2`, `_v2-pilot-ailow`, `_v2-full`, `_v2-1ep`, `_v2-150k607`, `_v2-ailow10`), `v2/raid_v2.jsonl`, `v2/raid_v2_meta.json`, `v2/news_pairs.jsonl`, `v2/news_pairs_meta.json` |
| `Yashhh999/gradientTrace-v2` | 150k / 3 epochs (5,265 steps) — fails the gate |
| `Yashhh999/gradientTrace-v2-1ep` | 150k / 1 epoch (1,755 steps) — fails the gate |
| `Yashhh999/gradientTrace-v2-150k607` | 150k at the pilot's 607 steps — fails the gate |
| `Yashhh999/gradientTrace-v2-ailow10` | 150k / 607 steps with `local_jsonl.ai_foundational` 0.10 — clears FPR@0.5, fails TPR@1% |
| `Yashhh999/gradientTrace-v2-pilot` | **the release model** (ailow: 50k / 607 steps) |
| `Yashhh999/gradientTrace-smoke` | smoke pushes only |

`Yashhh999/gradientTrace` and `-v2-pilot` are public (checked 2026-11-01); `-v2` is private
(anonymous 401). The cells still authenticate for writes. A training run pushes the best
model, `eval_report.json`, `data_manifest.json`, and a generated model card. With
`checkpoint_strategy: checkpoint`, each save replaces `last-checkpoint/` (optimizer + scheduler
state, ~4.4–5 GB per push).

---

## 10. Debugging playbook

| symptom | likely cause | action |
|---|---|---|
| `ImportError: attempted relative import` | module file run outside package | `python -m gradienttrace.train` / import via package |
| 401 from the Hub | token rotated after the leak | refresh `HF_TOKEN`, re-run |
| torchrun rank 0 exit 1, traceback scrolled away | exception inside the child | the Kaggle cell now reprints the traceback; read the block above `Failures:` |
| `create_repo` 403 / preflight fails | token cannot create or write the repo | create it once at huggingface.co/new (model, private) or use a write token with repo creation |
| `git push` hangs silently | Windows GCM runs before gh helper | push with `-c credential.helper=` reset (§2) |
| resume starts from step 0 | `RESUME=False`, or no complete checkpoint | set `RESUME=True`; check `last-checkpoint/` on the Hub |
| resumed run trained on the wrong recipe | data mix changed while resuming | data changes invalidate resume; start with `RESUME=False` (first save overwrites `last-checkpoint/`) |
| run ends at ~1 epoch | step evals are patience ticks | `early_stopping_patience: 0`; raise `SAVE_EVERY` |
| `test_humanized` empty | no humanized data (RAID not loaded / no local data) | add RAID or `variant: humanized` rows |
| `test_heldout_domain` empty | held-out domain absent from sources (poetry has no rows) | rebuild the RAID slice; check manifest |
| RAID slice has almost no humans | stale pre-fix slice (the first one had 1,233 in 3 domains) | rebuild with `build_raid.py`; the cell aborts below 10k humans or 7 domains |
| RAID domain has 0 rows | wrong byte window | do not mark complete; re-probe the window |
| eval is silent for hours | fp32 full eval (~3h on one T4) | use `finish_eval.py` (fp16, per-slice progress, Hub partials) |
| OOM at the first `optimizer.step()`, message names another PID | the notebook kernel still holds an in-kernel gate/eval model on GPU 0 (it survives the cell) | restart the session; killing torchrun does **not** free the kernel; the cell aborts if GPU 0 has <13 GiB free |
| GPU busy after restart, no free memory | orphaned torchrun from a crashed cell | `pkill -f torch.distributed.run`; `pkill -f kaggle_launch.py`; delete `runs/` |
| eval OOM | batch too large | gate/finish halve the batch automatically; training needs a smaller micro batch |
| TPU child exits 1 silently | kernel imported torch / torch_xla present | never import torch in the TPU kernel; remove torch_xla; restart session |
| `PjRt … premapped_buffer_size` | libtpu too old (torch_xla pin) | restore the libtpu `torch_tpu` declares |
| torch_tpu “no extension module compatible” | torch outside 2.11–2.15 | reinstall a supported torch build (cell handles 2.14.1) |
| internal AUROC 0.9999 but humans flagged | source confounding | ignore internal numbers; run the gate |
| gate numbers equal the v1 baseline / report `model` is `Yashhh999/gradientTrace` | `MODEL_ID` left at the default while `MODEL_LABEL` was changed | set `MODEL_ID` to the model under test; the cell now aborts on this |
| gate flags ~97% of humans as AI; pooled AUROC ~0.5-0.6 | a source's labels are inverted (MAGE ships 1 = human, 0 = machine) | use `mage`, not `hf:yaful/MAGE`; check every new source's label convention |
| gate ~0.96 on reviews but ~0.48 on news, 97% of news humans flagged | training humans are perturbed in news/reddit/books/poetry, clean only in reviews/wiki/recipes; AI is clean everywhere | train on the paired clean-news set (`colab/build_news.py`); it is downloaded automatically into `data/local.jsonl` |
| gate news human mean rises each time short human news is added (0.375 -> 0.510 -> 0.593) | short human rows displace the gate's long headline+wire form inside the fixed local human quota; the shift shows in both the 0–400 and 400–800 buckets | keep human news long (200–520): `build_news.py` now drops short human rows and keeps short AI variants only |
| `paired clean news missing` when the training cell starts | `v2/news_pairs.jsonl` has not been built/pushed, or `NEWS_HUB` is wrong | run `colab/build_news.py` (it appends/pushes partials), or point `LOCAL_JSONL` at your own paired file |
| `data/local.jsonl is required by source_shares but loaded no … rows` at data build | the pasted training cell is stale (skips the news download) or `LOCAL_JSONL` points nowhere | re-paste the current `one_shot_kaggle.py`; the build now aborts before training instead of silently using the old mix |
| news builder ends with one generator | a model architecture is unsupported by the installed transformers | read the printed load error; the cell falls back through `GEN_MODELS`; one generator is still usable, two are better |
| news builder crashes with `probability tensor contains either inf, nan or element < 0` | a generator is numerically unstable in fp16 on the T4 (Granite-2B) | use SmolLM2 (current list); do not add fp16-unstable models |
| news builder keeps only ~14% of generations | completions end below the 200-char loader minimum | read `[news] rejects:`; `min_new_tokens` now forces roughly the human length; if `preamble` dominates, strip the lead-in |
| fusion AUROC jumps but only on the rows it was fit against | the k-NN bank is the evaluation set; out-of-fold blocks labels, not clusters | measured 2026-11-03 (§7.9): the gain does not survive. Re-run with `colab/hybrid_bank_eval.py` before quoting any fusion number |
| k-NN fusion beats the gate-fitted baseline but loses to the bare detector on fresh data | embedding geometry is bank-local: nearest-neighbour distance only means something relative to the population in the bank | ship `det + stylometry`; treat vector features as diagnostic, not production |
| a fusion feature has a negative coefficient on `det_logit` | the embedding geometry carries the signal and the logit acts as a corrector; sign agreement was 5/5 folds | check sign stability per fold (`hybrid_eval.py` prints it); do not "fix" the sign |
| k-NN features look suspiciously strong | the gate is prompt-paired, so a row's nearest neighbour is its own AI/human partner — that is topic, not style | never compute vector features with group blocking disabled; `tests/test_hybrid.py` fails if it is |
| hybrid fusion has better AUROC and FPR@0.5 but reviews TPR@1% collapses | `log_chars` in the feature set; the fusion learns the gate's length confound | keep length out of the feature set; never ship the all-features fusion |
| every 150k build flags 60-71% of gate humans at 0.5 while the 50k pilot flags 24.75% | data volume, not steps: 607 / 1,755 / 5,265 steps all land at human mean 0.62-0.68 | the pilot is the release model; the only lever that moved it was `local_jsonl.ai_foundational`, and it trades the AI tail away to get it |

---

## 11. Current status (2026-11-03)

Compact state of the work. The dated narrative behind every number here — all eleven pilot
gates, the four 150k gates, and each diagnosis — is in `CHANGELOG.md`.

- **Release model:** `Yashhh999/gradientTrace-v2-pilot` (ailow mix: 50k samples, 1 epoch, 607
  steps). Gate: news 0.9487 / TPR@1% 0.7925 / FPR@0.5 0.2475, human mean 0.2623; reviews
  0.9360 / 0.3650 / 0.1375; pooled 0.9323 / 0.5887 / 0.1925. It is the only build that clears all
  four bars.
- **The 150k data-mix search is closed.** Four gates (5,265 / 1,755 / 607 steps at the v2 mix,
  then 607 steps with `local_jsonl.ai_foundational` 0.10) all fail: the step budget is not the
  variable, and the one weight that fixes clean-human calibration destroys the AI tail.
- **Hybrid shipped, and re-tested honestly** (`gradienttrace/hybrid.py`, `colab/hybrid_eval.py`,
  `colab/hybrid_bank_eval.py`): detector logit + 33 stylometric features + k-NN/centroid geometry
  in the detector's own pooled embedding, fused by out-of-fold logistic regression. Gate-fitted it
  claims pooled AUROC 0.9876 / TPR@1% 0.7688 and 9/20 frontier explainers — that run used the gate
  as its own bank. Under a bank drawn from the pilot's held-out slices (§7.9) the k-NN sets fall
  **below** the bare detector (pooled AUROC 0.9105 vs 0.9323, reviews TPR@1% 0.2125 vs 0.3650) and
  explainers return to 1/20, while stylometry holds: pooled AUROC **0.9422**, news TPR@1%
  **0.8050**, news FPR@0.5 **0.0825** (from 0.2475), reviews AUROC **0.9494** / FPR@0.5 **0.0450**.
  **Shipped: `SHIP_SET="det + stylometry"`** — no bank needed at inference.
- **Not yet done:** the fused model has never been through `colab/external_gate.py` as a deployed
  detector (the §7.9 numbers come from re-scoring the frozen gate with the shipped features), and
  `SHIP_SET` has not been validated on a second generator. Both precede any launch claim that
  uses fusion numbers rather than the pilot's recorded gate row.
- **Coverage gap that no scorer fixes:** encyclopedic/explainer register. Wikipedia's GBR lead
  scores 0.9695 as AI under every variant tried (detector, GPT-2 fusion, k-NN fusion). Fix is
  paired training data in that register, not a second model.

- **Root cause of the gate failure (found 2026-10-31):** `yaful/MAGE` labels 1 = human and
  0 = machine, while the generic `hf:` adapter assumes 1 = AI. `hf:yaful/MAGE` therefore fed
  ~5.6k clean human texts (cmv/eli5/…) into the AI bucket and ~5.6k machine texts into the
  human bucket in both v1 and the pilot; its `src` (domain + generator) was ignored. This is
  the "clean prose = AI" prior behind the gate's 96.75% human false-positive rate.

- **Fix:** a dedicated `mage` loader flips the labels, parses `src` into domain/generator,
  excludes the yelp split (the gate's reviews source), shuffles the train split so one domain
  cannot fill the pool, and `qwen` is held out (RAID ships qwen-1.5, the gate uses Qwen2.5).
  `configs/{base,v2,pilot}.yaml` now list `mage`.

- **v1 model:** trained to step 5,607 (~1 epoch, early-stopped), on the Hub. Internal eval:
  val 0.99998 / test_indomain 0.99995 / OOD 0.99999 AUROC, val threshold 0.9993,
  `test_humanized` empty. A probe scored unseen human ag_news at mean P(AI) 0.9553
  (95% above 0.9) and Yelp at 0.9950 (99.5% above 0.9). **Not shippable**, and its gate
  numbers are partly explained by the MAGE inversion; not a clean baseline for the recipe.

- **v2 data:** RAID slice healthy (71,297 rows; 18,848 humans in 7 domains; poetry held out).
  `configs/v2.yaml` and `configs/pilot.yaml` use hc3 + mage + arena + RAID + `local_jsonl`,
  source quotas, grouped splits, humanized cap, held-out poetry; `local_jsonl` carries the
  clean-news pairs.

- **Full-run bar:** hold news FPR@0.5 <=0.36, TPR@1% >=0.6, AI mean >=0.9, reviews TPR@1%
  >=0.30. If the bigger run regresses the human side, the next lever is not more clean humans
  (longhuman2) but the local AI weight and the short-variant cap.

- Do **not** retrain on the v1 mix, and do not resume the v1 checkpoint into v2 (different data).

---

## 12. Conventions for agents

- **Framing:** the user wants a working detector for late-2026 generators. Default to
  DeBERTa-v3-large and the TPU/2×T4 paths. Do not propose bigger models or reopen hardware
  choices unless asked.
- **Communication:** verdict/decision first; then the exact cell edits or commands; then what to
  check in the next log. Keep cells paste-ready (single cell, self-contained). When a run fails,
  ask for the traceback rather than guessing, but make the next run self-diagnosing.
- **Cost awareness:** the user pays for multi-hour sessions. Fail fast, checkpoint often, prefer
  resumable flows, and never silently discard a checkpoint.
- **Changes that touch data:** bump the config/Hub target, set `RESUME=False`, and verify
  `manifest["splits"]`, `manifest["train_buckets"]`, `manifest["train_sources"]`, and the
  OOD/humanized/held-out slice sizes. A data change invalidates every existing checkpoint.
  Check a new source's label convention before wiring it in; MAGE ships 1 = human, not 1 = AI.
- **Changes that touch the cell:** keep `colab/*.py` importable/compilable, update this file, and
  remember the notebook clones GitHub — push before telling the user to run.
- **Do not** print tokens, disable TLS checks, add `--no-verify`, or commit artifacts.

---

## 13. Quick reference

```bash
# pipeline sanity (network-free, ~40 steps)
python -m gradienttrace.train --config configs/base.yaml --smoke --max-steps 40

# local tests / CI equivalent
python -m compileall -q gradienttrace colab tests && python tests/test_metrics.py

# standalone eval of a checkpoint
python -m gradienttrace.evaluate --model runs/gradientTrace --config configs/base.yaml

# push (helper reset required)
GIT_TERMINAL_PROMPT=0 git -c credential.helper= \
  -c credential.helper='!gh auth git-credential' push origin main
```

Key constants: backbone `microsoft/deberta-v3-large`; `max_length=512`; global batch 64
(2×T4: 4×8×2; TPU: 4×16); LR 1.5e-5; warmup 10% + cosine; weight decay 0.01; head dropout 0.1;
labels 0=human, 1=AI; threshold from val Youden index; private Hub repos.
