# gradientTrace

Encoder-only AI-text detector built for **production-grade robustness**, not just
in-domain accuracy. One fine-tuning pipeline that runs unchanged on a **TPU
v5e-1** (bf16 + torch_xla) or **2x T4** (fp16 + DDP), with checkpointing straight
to the Hugging Face Hub.

Backbone: `microsoft/deberta-v3-large` (435M) - disentangled attention, best-in-class
text classification. ~435M parameters is deliberate: beyond this size you buy
overfitting to lexical tells, not robustness.

Compatibility: `transformers` 4.44+ and 5.x (Trainer/TrainingArguments differences
are detected at runtime). The full pipeline is smoke-tested end to end on
transformers 5.22 / torch 2.15; CUDA and TPU runs use the same code path.

## Why the data, not the model size, decides success

Robustness to unseen generators and paraphrasing comes from four things, all enforced here:

1. **Prompt-paired data** - human and AI text share a prompt/topic, so the model
   can't learn "Wikipedia vs Llama-2" artifacts instead of "human vs machine".
2. **Generator-disjoint evaluation** - generators listed in
   `data.held_out_generators` never enter training; they form the OOD test split.
3. **Humanized-text slice** - paraphrased AI text (variant `humanized`) is capped
   in training and isolated for evaluation. This is the attack that breaks every
   encoder detector.
4. **Sliced metrics** - AUROC and TPR@1%FPR per generator and domain, not one
   aggregate number.

## Architecture

```
text (<=512 tok)
  -> tokenizer (sentencepiece, DeBERTa-v3 vocab)
  -> encoder: DeBERTa-v3-large (24 layers, hidden 1024, disentangled attention)
  -> pool [CLS] (or mean, configurable)
  -> dropout 0.1
  -> Linear(1024 -> 1)
  -> sigmoid -> P(machine-generated)
loss = BCEWithLogitsLoss(labels: 0=human, 1=AI)
```

`AIDetector` is a `PreTrainedModel` whose `config_class` is `AutoConfig`, so
swapping the backbone family (e.g. `answerdotai/ModernBERT-large` for 8k-token
documents) is a config change, not a code change.

## Data recipe

| bucket | weight | sources | purpose |
|---|---|---|---|
| `human` | 0.40 | HC3 human answers, `artem9k/ai-text-detection-pile` (human rows), your own paired data | natural, variable human writing |
| `ai_foundational` | 0.40 | HC3 ChatGPT answers, pile model rows, any `hf:` source | baseline machine signature |
| `ai_frontier` | 0.20 | **your own prompt-paired API outputs in `data/local.jsonl`** | modern reasoning-model patterns |

The trainer loads what exists, logs what failed, and warns loudly if the frontier
slice is missing. You can add any Hub dataset with `hf:org/name:split` (columns
are auto-detected: text/generation, label/is_ai, model/generator, prompt/question).

### `data/local.jsonl` schema (frontier + humanized data)

```json
{"text": "human answer to prompt P", "label": 0, "generator": "human", "prompt": "P", "domain": "essay"}
{"text": "model answer to prompt P", "label": 1, "generator": "gpt-5", "prompt": "P", "domain": "essay"}
{"text": "humanized/paraphrased model answer to P", "label": 1, "generator": "gpt-5", "prompt": "P", "variant": "humanized"}
```

Pair every AI output with a human answer to the **same prompt**. Unpaired corpora
inflate in-domain scores and collapse under domain shift.

## Quickstart

### Local

```bash
pip install -r requirements.txt
python -m gradienttrace.train --config configs/base.yaml --smoke --max-steps 40   # pipeline check, ~minutes
python -m gradienttrace.train --config configs/base.yaml --push-to-hub           # real run
python -m gradienttrace.evaluate --model runs/gradientTrace --config configs/base.yaml
```

### One-shot Colab cell (GPU or TPU)

```python
RUN = "smoke"                        # "smoke" | "full"
HUB_ID = "Yashhh999/gradientTrace"
PUSH_SMOKE = False
RESUME = False

import os, shutil, subprocess, sys

try:
    from google.colab import userdata
    for key in ("HF_TOKEN", "WANDB_API_KEY"):
        try:
            value = userdata.get(key)
            if value:
                os.environ[key] = value
        except Exception:
            pass
except ImportError:
    pass

is_tpu = bool(os.environ.get("COLAB_TPU_ADDR") or os.environ.get("PJRT_DEVICE") == "TPU")

REPO_DIR = "/content/gradientTrace"
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    if os.path.isdir(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    subprocess.run(f"git clone --depth 1 https://github.com/Yashhh999/gradientTrace.git {REPO_DIR}", shell=True, check=True)
else:  # keep runs/ so RESUME works
    subprocess.run(f"git -C {REPO_DIR} pull --ff-only", shell=True, check=True)
os.chdir(REPO_DIR)
subprocess.run("pip -q install -r requirements.txt", shell=True, check=True)

if is_tpu:
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    from importlib.metadata import PackageNotFoundError, version
    torch_base = version("torch").split("+")[0]
    try:
        xla_base = version("torch_xla").split("+")[0]
    except PackageNotFoundError:
        xla_base = None
    if xla_base != torch_base:   # _XLAC.so is ABI-locked to one torch build
        print(f"torch_xla {xla_base} != torch {torch_base}; aligning")
        result = subprocess.run(
            f"pip -q install 'torch_xla[tpu]=={torch_base}' -f https://storage.googleapis.com/libtpu-releases/index.html",
            shell=True,
        )
        if result.returncode != 0 and xla_base:
            result = subprocess.run(f"pip -q install 'torch=={xla_base}'", shell=True)
        if result.returncode != 0:
            subprocess.run("pip -q install 'torch_xla[tpu]' -f https://storage.googleapis.com/libtpu-releases/index.html", shell=True, check=True)
        torch_base, xla_base = version("torch").split("+")[0], version("torch_xla").split("+")[0]
        if xla_base != torch_base:
            print(f"WARNING: torch {torch_base} vs torch_xla {xla_base} still differ")
    print(f"torch {version('torch')} | torch_xla {version('torch_xla')} | tpu=True")
else:
    import torch, transformers
    assert torch.cuda.is_available(), "Select a GPU or TPU runtime."
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | tpu=False")

push = RUN == "full" or PUSH_SMOKE
hub = HUB_ID if RUN == "full" else HUB_ID + "-smoke"
cmd = [sys.executable, "-m", "gradienttrace.train", "--config", "configs/base.yaml", "--hub-model-id", hub]
if RUN == "smoke":
    cmd += ["--smoke", "--max-steps", "40"]
if push:
    assert os.environ.get("HF_TOKEN"), "Add HF_TOKEN to Colab Secrets."
    cmd.append("--push-to-hub")
if RESUME:
    cmd.append("--resume")
subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
```

This file is also at [`colab/one_shot.py`](colab/one_shot.py). Two Colab-TPU
constraints shape the cell: `torch_xla`'s `_XLAC` extension must be the build
compiled for the installed torch (otherwise it dies on `undefined symbol`), and
the kernel must not import torch on TPU, because that initializes libtpu in the
kernel and the training subprocess can then no longer claim the chip. The smoke
run is network-free (synthetic paired data) and verifies data, training,
evaluation, and optional Hub upload end to end before you spend GPU/TPU hours.

## Training configuration (defaults)

| setting | value | reason |
|---|---|---|
| LR | 1.5e-5, 10% linear warmup + cosine | encoder fine-tuning stability |
| weight decay | 0.01 | regularization |
| head dropout | 0.1 | regularize the classifier, not the pretrained encoder |
| max_length | 512 | DeBERTa-v3 position limit |
| batch | 8x8 (TPU) / 16x4 (T4) = 64 | fits 16GB, DDP/XLA-safe |
| precision | bf16 on TPU/Ampere+, fp16+GradScaler on T4 | T4 (Turing) has no bf16 |
| eval/save | per epoch, keep 2 | `load_best_model_at_end` on val AUROC |
| early stopping | patience 2 on AUROC | the real anti-overfitting lever |

Device handling: `gradienttrace/runtime.py` picks static padding + 0 dataloader
workers on XLA (avoids recompiles and forked-worker issues), dynamic padding +
grouped lengths on CUDA. DDP on 2 GPUs is plain `torchrun`; no DeepSpeed/FSDP
for a 435M model at 512 tokens.

## Evaluation protocol

`evaluate.py` (and the end of every training run) reports, per slice:
AUROC, AUPRC, accuracy@0.5, **TPR@FPR=1% and 5%**, calibration (ECE), plus
per-generator and per-domain breakdowns.

| slice | what it tells you |
|---|---|
| `test_indomain` | seen generators, unseen prompts - the easy number |
| `test_ood_generators` | held-out model families - the number that matters |
| `test_humanized` | paraphrased machine text - the adversarial proxy |
| `custom` (`--eval-jsonl`) | any fresh data, e.g. this month's frontier outputs |

A detector is only as good as its OOD and humanized columns. Ship nothing that
has not been measured there.

## Hugging Face Hub

Every checkpoint save is pushed to `hub.model_id` (`hub_strategy: every_save`),
plus the final best model, `eval_report.json`, `data_manifest.json`, and an
auto-generated model card with the held-out table. Set `HF_TOKEN` (write scope).
For minimal uploads, set `hub.checkpoint_strategy: end`.

## Hardware notes

| | 2x T4 | TPU v5e-1 |
|---|---|---|
| precision | fp16 + GradScaler | bf16 native |
| memory | 16GB per card (full replica) | 16GB, single device |
| sync | PCIe all-reduce each step | none |
| quirks | no bf16/TF32 | torch_xla install pins torch version |

Recommended: v5e-1. If you use the T4s, prefer two parallel single-GPU experiments
over DDP - iteration speed beats one slightly faster job.

## Limitations

- Scores are evidence, not proof. Expect false positives on atypical human prose.
- Humanized/paraphrased machine text remains the hard failure mode; the
  `test_humanized` slice exists to keep that honest, not to claim it is solved.
- A detector trained today degrades against next quarter's generators. Keep the
  `ai_frontier` slice fresh and retrain on a schedule; keep an eval set that
  never enters training.
- For product use, complement with watermarking/provenance where available.

## Repo layout

```
gradienttrace/
  config.py     config dataclasses + YAML loader
  runtime.py    CUDA / XLA / CPU detection, precision + batch defaults
  data.py       paired sources, dedup, held-out routing, smoke data
  model.py      AIDetector (PreTrainedModel) + tokenizer helpers
  metrics.py    AUROC/AUPRC/TPR@FPR/ECE, sliced reports
  train.py      HF Trainer pipeline, early stopping, Hub checkpoints
  evaluate.py   standalone sliced evaluation
configs/base.yaml
colab/one_shot.py
tests/test_metrics.py
data/example_local.jsonl   schema example (copy to data/local.jsonl)
```

## License

MIT
