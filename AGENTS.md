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
- **Current phase (2026-10-31):** v2 data work is done (RAID slice rebuilt and healthy);
  the 50k one-epoch pilot has not yet completed a successful training run. The external
  gate on v1 is the recorded baseline. Do not start the full 150k v2 run until the pilot
  passes the gate.

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
  train.py      HF Trainer pipeline, resume, Hub checkpoints, final slice eval
  evaluate.py   standalone sliced evaluation, predict_probs
configs/
  base.yaml     v1 recipe (hc3 + pile + MAGE + arena)
  v2.yaml       paired recipe + RAID + held-out domain poetry
  pilot.yaml    50k / 1 epoch pilot of the v2 recipe
colab/
  one_shot.py         Colab cell (GPU or TPU v5e-1), base config only
  one_shot_kaggle.py  Kaggle 2x T4 cell (CONFIG knob, RAID download, Hub preflight)
  external_gate.py    external acceptance gate (frozen matched human/AI dataset)
  finish_eval.py      resumable full slice evaluation on a single GPU
  build_raid.py       one-time RAID byte-range extractor -> Hub v2/raid_v2.jsonl
data/example_local.jsonl  schema example for prompt-paired `local.jsonl`
tests/test_metrics.py     metric unit tests
.github/workflows/ci.yml  compileall + metric tests
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
| `pile` | `artem9k/ai-text-detection-pile` | `source` column only distinguishes human vs ai; no generator, no prompt. v1 only — **not in v2/pilot** |
| `arena` | `lmarena-ai/arena-human-preference-55k` | `model_a/b`, `response_a/b` (JSON-list strings per turn); only public generator-labelled frontier source; domain `chat` |
| `hf:<org/name>[:split]` | generic adapter | auto-detects text/label/generator/prompt/domain columns; MAGE tops up `ai_foundational` |
| `raid` | `cfg.data.raid_jsonl` | paired human/model rows, `variant` raw/humanized, `attack`, `group`; see 5.5 |
| `local_jsonl` | `cfg.data.local_jsonl` | user-supplied paired data; see schema below |

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
| hf:yaful/MAGE | 0.35 | 0.30 | — |
| raid | 0.40 | 0.45 | 0.80 |
| arena | — | — | 0.40 |
| local_jsonl | — | — | — |

Behavior (`_select_by_share`):

- Weights are normalized across sources that actually have samples in that bucket.
- Quota = `min(pool_size, round(target * weight / total_weight))`.
- Any shortfall is redistributed to sources with spare capacity (up to 10 rounds).
- While loading, each source pool is capped at `int(target*share*1.5) + 64` to leave headroom.
- A source with no `source_shares` entry only fills shortfalls (warning logged).
- If no weights are usable, sources split the bucket evenly.

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
| sources | hc3, pile, MAGE, arena, local | hc3, MAGE, arena, raid, local | same as v2 |
| shares | none | per 5.3 | per 5.3 |
| RAID | no | `data/raid_v2.jsonl` | same |
| held-out domain | — | poetry | poetry |
| output dir | `runs/gradientTrace` | `runs/gradientTrace-v2` | `runs/gradientTrace-v2-pilot` |
| Hub repo | `Yashhh999/gradientTrace` | `…-v2` | `…-v2-pilot` |
| early stopping | patience 0 | 0 | 0 |
| length match | — | false | false |

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
  1,233 humans in three domains and would otherwise train silently).
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
- AI: `Qwen/Qwen2.5-3B-Instruct` (unseen in training) writes one response per human sample,
  using the human's snippet as the topic and matching its length; sampling T=0.8, top_p=0.9,
  batch 8.
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
(650), wiki (293), recipes (290); all other human rows are perturbed — a known caveat when
interpreting style cues.

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
- When asked about a live run, answer verdict-first (healthy / not healthy), name only real
  problems, and mention known-harmless noise in one line. State decisions (stop/resume) first.

Known-harmless log noise: duplicate per-rank lines, NCCL `barrier()`/`destroy_process_group()`
warnings, tokenizer PAD/BOS/EOS alignment notes, LOAD REPORT `UNEXPECTED` MLM heads, brief
step-time spikes at checkpoint saves.

---

## 9. Hub artifacts

| repo | contents |
|---|---|
| `Yashhh999/gradientTrace` | v1 model at root + `checkpoint-5607/`, `last-checkpoint/`, `eval_report.json`, `data_manifest.json`, README, `external_gate_v1.jsonl`, `external_gate_v1_meta.json`, `external_gate_v1_report_v1.json`, `v2/raid_v2.jsonl`, `v2/raid_v2_meta.json` |
| `Yashhh999/gradientTrace-v2` | target for the full v2 run |
| `Yashhh999/gradientTrace-v2-pilot` | target for the 50k pilot |
| `Yashhh999/gradientTrace-smoke` | smoke pushes only |

Repos are private (`hub.private: true`); anonymous access is 401. A training run pushes the
best model, `eval_report.json`, `data_manifest.json`, and a generated model card. With
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

---

## 11. Current status (2026-10-31)

- **v1 model:** trained to step 5,607 (~1 epoch, early-stopped), on the Hub. Internal eval:
  val 0.99998 / test_indomain 0.99995 / OOD 0.99999 AUROC, val threshold 0.9993,
  `test_humanized` empty. A probe scored unseen human ag_news at mean P(AI) 0.9553
  (95% above 0.9) and Yelp at 0.9950 (99.5% above 0.9). **Not shippable.**
- **External gate v1:** pooled AUROC 0.5270, TPR@1% FPR 0.0000, FPR@0.5 0.9750 (§7.5 table).
  This is the honest baseline for v2.
- **v2 data:** RAID slice rebuilt and healthy (71,297 rows; 18,848 humans in 7 domains; poetry
  held out).
  `configs/v2.yaml` and `configs/pilot.yaml` use hc3 + MAGE + arena + RAID, source quotas,
  grouped splits, humanized cap, held-out poetry.
- **Pilot (2026-10-31):** completed. 607 steps (1 epoch at global 64), 4,637 s of training,
  fp16 on 2×T4; val AUROC 0.9957 at step 300 and 0.9972 at step 600 (TPR@1% 0.9498). Best
  model + `eval_report.json` + `data_manifest.json` + README pushed to
  `Yashhh999/gradientTrace-v2-pilot`. Final slice eval: val 5,089 → 0.9972/0.950;
  test_indomain 4,564 → 0.9946/0.913; test_ood_generators 11,900 → 0.9986/0.982;
  test_humanized 756 → 0.9960/0.955; test_heldout_domain 5,060 → **0.7512/0.172**. The
  held-out domain (poetry) is the weak spot; internal numbers are still not detection quality.
- **Pilot data build (same log):** train 38,792 / val 5,089; slices 4,564 in-domain, 11,900
  OOD, 756 humanized, 5,060 held-out domain. Buckets 16,100 human / 15,989 foundational /
  6,703 frontier — about 8.4k of the 10k frontier target (arena's per-source cap of 6,064 plus
  RAID's 2,381 frontier rows). At 150k the 30k frontier target can only reach ~20.4k from the
  same sources; raise arena's frontier share or rebuild RAID with higher caps after the gate.
- **Next steps, in order:**
  1. Gate the pilot with the frozen dataset: `MODEL_ID="Yashhh999/gradientTrace-v2-pilot"`,
     `MODEL_LABEL="v2-pilot"`, `GATE_NAME="external_gate_v1"`, `REGENERATE=False`, `PUSH=True`.
  2. Compare pooled TPR@1% FPR / AUROC against the v1 baseline (0.0000 / 0.5270). A TPR still
     at 0 means the data change did not help; a clear move means proceed. Fix the frontier
     supply before the full run.
  3. If the gate passes, run the full 150k v2 recipe (`Yashhh999/gradientTrace-v2`, ~12h on
     2×T4, ~30h on v5e-1) and gate that model the same way.
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
