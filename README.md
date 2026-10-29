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
# gradientTrace - one-shot Colab cell
# Paste this whole file into a single Colab cell.
# Runtime: GPU (T4/L4/A100) or TPU v5e-1.
# Colab Secrets: HF_TOKEN (write scope) required for Hub checkpoints; WANDB_API_KEY optional.
RUN = "smoke"                        # "smoke" (~10 min sanity) | "full" (real training)
HUB_ID = "Yashhh999/gradientTrace"   # HF model repo to receive checkpoints
PUSH_SMOKE = False                   # also push the smoke model to "<HUB_ID>-smoke"
MAX_STEPS = 600                      # full run: stop after this many optimizer steps per session
                                     # (~2.5-3 h on v5e-1; RESUME keeps counting from the checkpoint)
                                     # 0 = all EPOCHS, bounded only by Colab's session limit
SAVE_EVERY = 300                     # full run: checkpoint + eval every N steps, pushed to the Hub
EPOCHS = None                        # None = config (3); e.g. 1 for a single full-data pass
RESUME = False                       # continue from the newest checkpoint: local first, else the Hub

import os, shutil, subprocess, sys


def run(command, check=True):
    """Stream a child's output through the kernel; Colab sometimes hides raw child fds."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with subprocess.Popen(
        command,
        shell=isinstance(command, str),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    ) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
    if check and proc.returncode:
        raise RuntimeError(f"command failed with exit {proc.returncode}: {command}")
    return proc.returncode


try:  # Colab secrets -> env
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
    subprocess.run(
        f"git clone --depth 1 https://github.com/Yashhh999/gradientTrace.git {REPO_DIR}",
        shell=True,
        check=True,
    )
else:  # keep runs/ so RESUME works
    subprocess.run(f"git -C {REPO_DIR} pull --ff-only", shell=True, check=True)
os.chdir(REPO_DIR)
run("pip -q install -r requirements.txt")

if is_tpu:
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    from importlib.metadata import PackageNotFoundError, version

    def installed(name):
        try:
            return version(name).split("+")[0]
        except PackageNotFoundError:
            return None

    # torch_xla is a stale leftover in this image. While it is installed,
    # transformers and accelerate take the XLA code path instead of the native
    # TPU backend, and its _XLAC.so is ABI-locked to a torch build that no
    # longer exists here (undefined symbol on import).
    had_xla = installed("torch_xla") is not None
    run("pip -q uninstall -y torch_xla", check=False)

    # Installing torch_xla also downgraded libtpu (2.8.0 pins 0.0.17), whose
    # PJRT plugin is too old for the image's torch_tpu client: creating the
    # PjRt client dies on "Unexpected option name ... premapped_buffer_size".
    # Restore the libtpu torch_tpu declares, else the newest published one.
    from importlib.metadata import requires
    from packaging.requirements import Requirement

    libtpu_req = None
    try:
        for raw in requires("torch_tpu") or []:
            req = Requirement(raw)
            if req.name.replace("_", "-").lower() == "libtpu":
                libtpu_req = req
                break
    except Exception:
        pass
    libtpu_v = installed("libtpu")
    if libtpu_req is not None and str(libtpu_req.specifier):
        if libtpu_v is None or not libtpu_req.specifier.contains(libtpu_v, prereleases=True):
            print(f"[gradientTrace] libtpu {libtpu_v} -> {libtpu_req.specifier} (required by torch_tpu)")
            run(f"pip -q install 'libtpu{libtpu_req.specifier}'", check=False)
    elif had_xla:
        print(f"[gradientTrace] torch_xla pinned libtpu {libtpu_v}; restoring the newest libtpu")
        run("pip -q install -U libtpu", check=False)

    # The native torch_tpu backend loads the compiled extension matching the
    # running torch; extensions ship for torch 2.11-2.15. 2.14.1 is the build
    # this image originally came with.
    supported = ("2.11", "2.12", "2.13", "2.14", "2.15")

    def torch_build():
        value = installed("torch")
        minor = ".".join(value.split(".")[:2]) if value else None
        return minor, value

    minor, torch_v = torch_build()
    if minor not in supported:
        print(f"[gradientTrace] torch {torch_v} has no torch_tpu extension; restoring 2.14.1")
        run("pip -q uninstall -y torchvision torchaudio", check=False)
        for target in ("2.14.1", "2.14.0"):
            run(
                f"pip -q install 'torch=={target}' --no-deps "
                "--index-url https://download.pytorch.org/whl/cpu",
                check=False,
            )
            minor, torch_v = torch_build()
            if minor == "2.14":
                break
    if minor not in supported:
        raise RuntimeError(
            f"torch {torch_v} cannot drive the native TPU backend (need 2.11-2.15); see pip output above"
        )

    # Do NOT import torch in this kernel: on TPU that claims the chip and the
    # training child can then no longer initialize libtpu. Verify in a
    # short-lived child instead: detection, a manual seed (the first call that
    # creates the PjRt client) and a real device op, so a bad libtpu fails here
    # instead of half-way through Trainer setup.
    try:
        run(
            [
                sys.executable,
                "-c",
                "from gradienttrace.runtime import detect_runtime;"
                "import torch; import importlib.metadata as md;"
                "r = detect_runtime();"
                "assert r.kind == 'tpu', f'expected the native TPU backend, got {r}';"
                "torch.manual_seed(0);"
                "total = float(torch.ones(8, device='tpu:0').sum());"
                "print('[gradientTrace]', r.name, '| precision', r.precision,"
                " '| micro_batch', r.micro_bs, 'x accum', r.grad_accum,"
                " '| libtpu', md.version('libtpu'), '| device op ok:', total)",
            ]
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}\nIf the error mentions a PjRt/libtpu mismatch, restart the runtime "
            "(Runtime > Restart session) and re-run this cell."
        ) from exc
    print(f"[gradientTrace] torch {torch_v} | native torch_tpu | torch_xla removed")
else:
    import torch
    import transformers

    assert torch.cuda.is_available(), "Select a GPU or TPU runtime: Runtime > Change runtime type."
    print(f"[gradientTrace] torch {torch.__version__} | transformers {transformers.__version__} | tpu=False")

# Optional: attach prompt-paired frontier/humanized data stored on Drive
#   shutil.copy("/content/drive/MyDrive/gradientTrace/local.jsonl", "data/local.jsonl")

push = RUN == "full" or PUSH_SMOKE
hub = HUB_ID if RUN == "full" else HUB_ID + "-smoke"
cmd = [sys.executable, "-m", "gradienttrace.train", "--config", "configs/base.yaml",
       "--hub-model-id", hub]
if RUN == "smoke":
    cmd += ["--smoke", "--max-steps", "40"]
else:
    if MAX_STEPS:
        cmd += ["--max-steps", str(MAX_STEPS)]
    if SAVE_EVERY:
        cmd += ["--save-steps", str(SAVE_EVERY)]
    if EPOCHS:
        cmd += ["--epochs", str(EPOCHS)]
if push:
    assert os.environ.get("HF_TOKEN"), "Add HF_TOKEN (write scope) to Colab Secrets to push checkpoints."
    cmd.append("--push-to-hub")
if RESUME:
    cmd.append("--resume")

print("[gradientTrace]", " ".join(cmd))
run(cmd)
print("[gradientTrace] done" + (f" -> https://huggingface.co/{hub}" if push else " (local run, no push)"))
```

This file is also at [`colab/one_shot.py`](colab/one_shot.py). Three Colab-TPU
constraints shape the cell: the native `torch_tpu` backend only loads a compiled
extension built for torch 2.11-2.15, a leftover `torch_xla` would take over
detection and drag in a libtpu too old for it (so the cell removes it and
restores the libtpu `torch_tpu` declares), and the kernel must not import torch
on TPU, because that initializes libtpu in the kernel and the training
subprocess can then no longer claim the chip. The smoke run is network-free
(synthetic paired data) and verifies data, training, evaluation, and optional Hub
upload end to end before you spend GPU/TPU hours.

A full 3-epoch run over 150k samples outlasts one Colab TPU session. `SAVE_EVERY`
checkpoints and evaluates every N optimizer steps and each save is pushed to the
Hub, so a killed VM loses at most that window; re-running the cell with
`RESUME = True` continues from the newest local checkpoint or, on a fresh VM,
from `last-checkpoint/` on the Hub (optimizer and scheduler state included).
`MAX_STEPS` optionally bounds a single session; on resume it counts additional
steps from the checkpoint.

### One-shot Kaggle cell (2x T4)

[`colab/one_shot_kaggle.py`](colab/one_shot_kaggle.py) is the same cell adapted to
Kaggle: notebook settings **Accelerator = GPU T4 x2**, **Internet = On**, and
`HF_TOKEN` (write scope) under **Add-ons > Secrets**. It installs requirements,
clones the repo into `/kaggle/working`, and launches
`python -m torch.distributed.run --standalone --nproc_per_node 2` so both T4s train
in DDP. Activation checkpointing is on, so bf16-less T4s run fp16 with a
GradScaler at 4 micro-batch x 8 accumulation per GPU = 64 global, matching the
TPU run. Warmup and throughput logging account for the extra process, and only
rank 0 evaluates, saves, and pushes to the Hub.

`RUN="smoke"` is a roughly ten-minute two-GPU pipeline check; `RUN="full"` trains,
with the same `MAX_STEPS` / `SAVE_EVERY` / `EPOCHS` / `RESUME` knobs as Colab.
Kaggle sessions are capped at 12 hours and the working directory does not persist,
so a full run spans a few sessions: push, then start a fresh notebook with
`RESUME = True` and the cell re-downloads `last-checkpoint/` from the Hub. Two T4s
at 64 global batch land around 1.7-1.9x a single T4, i.e. roughly 13-18 hours for
three epochs.

## Training configuration (defaults)

| setting | value | reason |
|---|---|---|
| LR | 1.5e-5, 10% linear warmup + cosine | encoder fine-tuning stability |
| weight decay | 0.01 | regularization |
| head dropout | 0.1 | regularize the classifier, not the pretrained encoder |
| max_length | 512 | DeBERTa-v3 position limit |
| batch | 4x16 (TPU) / 16x4 (T4) / 4x8x2 (2xT4) = 64 | fits the 16GB v5e-1, DDP/XLA-safe |
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
