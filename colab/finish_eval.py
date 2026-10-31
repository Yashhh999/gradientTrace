# gradientTrace - finish evaluation on a Colab T4 or Kaggle 2x T4 (paste this whole file into one cell)
# Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Accelerator > GPU T4 x2, Internet > On.
# Secrets: HF_TOKEN (write scope). Each slice is evaluated separately, saved to
# runs/gradientTrace/eval_partial.json and pushed to the Hub, so a disconnect never costs
# more than the slice in flight.
HUB_ID = "Yashhh999/gradientTrace"
BATCH_SIZE = 32          # fp16 eval; an OOM halves it automatically
PUSH = True              # upload eval_report.json, data_manifest.json, README.md
RESUME_PARTIAL = True    # skip slices already present in eval_partial.json

import importlib.util
import json
import logging
import os
import socket
import subprocess
import sys
import time

T0 = time.time()
REPO_DIR = "/kaggle/working/gradientTrace" if os.path.isdir("/kaggle/working") else "/content/gradientTrace"


def stage(message):
    print(f"\n[finish +{time.time() - T0:5.0f}s] {message}", flush=True)


token = os.environ.get("HF_TOKEN") or ""
if not token and os.path.isdir("/kaggle/working"):  # never getpass: it blocks the cell
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN") or ""
    except Exception as exc:
        print("[finish] Kaggle secret unavailable:", exc, flush=True)
if not token:
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN") or ""
    except Exception:
        pass
if not token:
    raise SystemExit(
        "[finish] HF_TOKEN missing. Kaggle: Add-ons > Secrets, add HF_TOKEN (write scope) and "
        "enable it for this notebook; Colab: key icon in the left sidebar. Then re-run."
    )
os.environ["HF_TOKEN"] = token
print("[finish] HF_TOKEN present (value not printed)", flush=True)

try:  # Kaggle silently blocks clone/HF downloads when Internet is off
    socket.create_connection(("huggingface.co", 443), timeout=10).close()
except OSError as exc:
    raise SystemExit(
        "[finish] no internet: Kaggle Settings > Internet > On (or Colab network is down), "
        "then re-run this cell."
    ) from exc

if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    stage(f"cloning into {REPO_DIR}")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/Yashhh999/gradientTrace.git", REPO_DIR],
        check=True,
    )
else:
    stage("updating the existing clone")
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only", "--quiet"], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(
    "[finish] repo at",
    subprocess.run(
        ["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip(),
    flush=True,
)

missing = [
    package
    for module, package in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("sentencepiece", "sentencepiece"),
        ("sklearn", "scikit-learn"),
        ("yaml", "pyyaml"),
        ("safetensors", "safetensors"),
        ("tqdm", "tqdm"),
    )
    if importlib.util.find_spec(module) is None
]
if missing:
    stage(f"installing missing packages: {missing}")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *missing], check=True)
else:
    print("[finish] dependencies already present; skipping pip", flush=True)

import torch  # noqa: E402  (only after the dependency check)

if not torch.cuda.is_available():
    raise SystemExit(
        "[finish] no GPU: Kaggle Settings > Accelerator > GPU T4 x2, or Colab "
        "Runtime > Change runtime type > T4 GPU, then re-run"
    )
import transformers  # noqa: E402

print(
    f"[finish] GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__}"
    f" | transformers {transformers.__version__}",
    flush=True,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", force=True)

from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

from gradienttrace.config import load_config  # noqa: E402
from gradienttrace.data import build_datasets  # noqa: E402
from gradienttrace.evaluate import evaluate_slices, load_model_and_tokenizer  # noqa: E402
from gradienttrace.runtime import detect_runtime, resolve_device  # noqa: E402

stage("building eval data (streams hc3 / pile / MAGE / arena; ~10-20 min)")
cfg = load_config("configs/base.yaml")
runtime = detect_runtime()
device = resolve_device(runtime)
print(
    f"[finish] runtime={runtime.name} precision={runtime.precision}"
    f" pad_to_max={runtime.pad_to_max} device={device}",
    flush=True,
)
bundle = build_datasets(cfg, smoke=False)
slices = {name: items for name, items in bundle.slices.items() if items}
print("[finish] slices:", {name: len(items) for name, items in slices.items()}, flush=True)

stage(f"loading {HUB_ID}")
model, tokenizer = load_model_and_tokenizer(HUB_ID)
model.to(device)
max_length = int(getattr(model.config, "gradienttrace_max_length", cfg.max_length))
print(
    f"[finish] params {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M"
    f" | max_length {max_length}",
    flush=True,
)

out_dir = Path("runs/gradientTrace")
out_dir.mkdir(parents=True, exist_ok=True)
partial_path = out_dir / "eval_partial.json"
api = None
if PUSH:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])

report = {}
if PUSH and RESUME_PARTIAL and not partial_path.is_file():
    try:  # a fresh VM starts empty; the previous session's slices are on the Hub
        from huggingface_hub import hf_hub_download

        fetched = hf_hub_download(
            repo_id=HUB_ID, filename="eval_partial.json", token=os.environ["HF_TOKEN"]
        )
        partial_path.write_text(Path(fetched).read_text(encoding="utf-8"), encoding="utf-8")
        print("[finish] restored eval_partial.json from the Hub", flush=True)
    except Exception as exc:
        print("[finish] no Hub partial to restore:", repr(exc), flush=True)
if RESUME_PARTIAL and partial_path.is_file():
    report = json.loads(partial_path.read_text(encoding="utf-8")).get("metrics", {})
    print("[finish] partial report found; done:", sorted(report), flush=True)


def eval_slice(name, batch_size):
    try:
        return evaluate_slices(
            model,
            tokenizer,
            {name: slices[name]},
            device,
            max_length=max_length,
            batch_size=batch_size,
            pad_to_max=runtime.pad_to_max,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if batch_size <= 8:
            raise
        print(f"[finish] OOM at batch {batch_size}; retrying {name} at {batch_size // 2}", flush=True)
        return eval_slice(name, batch_size // 2)


for name in slices:
    if name in report:
        print(f"[finish] {name}: already evaluated, skipping", flush=True)
        continue
    stage(f"evaluating {name} (n={len(slices[name])})")
    started = time.time()
    report.update(eval_slice(name, BATCH_SIZE))
    partial_path.write_text(
        json.dumps({"model": HUB_ID, "metrics": report}, indent=2, default=float), encoding="utf-8"
    )
    overall = report[name]["overall"]
    print(
        f"[finish] {name}: auroc={overall.get('auroc')}"
        f" tpr@1%fpr={overall.get('tpr_at_fpr_0.01')}"
        f" in {time.time() - started:.0f}s | total {time.time() - T0:.0f}s",
        flush=True,
    )
    if api is not None:
        try:
            api.upload_file(
                path_or_fileobj=str(partial_path),
                path_in_repo="eval_partial.json",
                repo_id=HUB_ID,
                commit_message=f"gradientTrace partial eval: {name}",
            )
            print("[finish] partial report pushed to the Hub", flush=True)
        except Exception as exc:
            print("[finish] partial push failed (eval continues):", repr(exc), flush=True)

final_report = {
    "hub_model_id": HUB_ID if PUSH else None,
    "runtime": asdict(runtime),
    "config": cfg.to_dict(),
    "splits": {
        name: len(items)
        for name, items in {"train": bundle.train, "val": bundle.val, **bundle.slices}.items()
    },
    "threshold": report.get("val", {}).get("overall", {}).get("best_threshold"),
    "metrics": report,
}
(out_dir / "eval_report.json").write_text(
    json.dumps(final_report, indent=2, default=float), encoding="utf-8"
)
(out_dir / "data_manifest.json").write_text(
    json.dumps(bundle.manifest, indent=2, default=float), encoding="utf-8"
)


def fmt(value, spec=".4f"):
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "n/a"


print(f"\n[finish] {'slice':22s} {'n':>6s} {'auroc':>8s} {'tpr@1%fpr':>9s}", flush=True)
for name, metrics in report.items():
    overall = metrics.get("overall", {})
    print(
        f"[finish] {name:22s} {overall.get('n', 0):6d}"
        f" {fmt(overall.get('auroc')):>8s} {fmt(overall.get('tpr_at_fpr_0.01'), '.3f'):>9s}",
        flush=True,
    )
print(f"[finish] threshold (val-calibrated): {final_report['threshold']}", flush=True)

if PUSH:
    from gradienttrace.train import _model_card

    for name in ("eval_report.json", "data_manifest.json"):
        api.upload_file(
            path_or_fileobj=str(out_dir / name),
            path_in_repo=name,
            repo_id=HUB_ID,
            commit_message=f"gradientTrace {name}",
        )
    api.upload_file(
        path_or_fileobj=_model_card(cfg, report, runtime).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HUB_ID,
        commit_message="gradientTrace model card",
    )
    print(
        f"[finish] pushed eval_report.json, data_manifest.json, README.md"
        f" -> https://huggingface.co/{HUB_ID}",
        flush=True,
    )

print(f"\n[finish] done in {time.time() - T0:.0f}s", flush=True)
