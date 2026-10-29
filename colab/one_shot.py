# gradientTrace - one-shot Colab cell
# Paste this whole file into a single Colab cell.
# Runtime: GPU (T4/L4/A100) or TPU v5e-1.
# Colab Secrets: HF_TOKEN (write scope) required for Hub checkpoints; WANDB_API_KEY optional.
RUN = "smoke"                        # "smoke" (~10 min sanity) | "full" (real training)
HUB_ID = "Yashhh999/gradientTrace"   # HF model repo to receive checkpoints
PUSH_SMOKE = False                   # also push the smoke model to "<HUB_ID>-smoke"
RESUME = False                       # resume from the last local checkpoint

import os, shutil, subprocess, sys

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
if os.path.isdir(REPO_DIR):
    shutil.rmtree(REPO_DIR)
subprocess.run(
    f"git clone --depth 1 https://github.com/Yashhh999/gradientTrace.git {REPO_DIR}",
    shell=True,
    check=True,
)
os.chdir(REPO_DIR)
subprocess.run("pip -q install -r requirements.txt", shell=True, check=True)

if is_tpu:
    print("[gradientTrace] TPU runtime detected: installing torch_xla")
    subprocess.run(
        "pip -q install 'torch_xla[tpu]' -f https://storage.googleapis.com/libtpu-releases/index.html",
        shell=True,
        check=True,
    )
    os.environ["PJRT_DEVICE"] = "TPU"

import torch
import transformers

if not is_tpu:
    assert torch.cuda.is_available(), "Select a GPU or TPU runtime: Runtime > Change runtime type."
print(f"[gradientTrace] torch {torch.__version__} | transformers {transformers.__version__} | tpu={is_tpu}")

# Optional: attach prompt-paired frontier/humanized data stored on Drive
#   shutil.copy("/content/drive/MyDrive/gradientTrace/local.jsonl", "data/local.jsonl")

push = RUN == "full" or PUSH_SMOKE
hub = HUB_ID if RUN == "full" else HUB_ID + "-smoke"
cmd = [sys.executable, "-m", "gradienttrace.train", "--config", "configs/base.yaml",
       "--hub-model-id", hub]
if RUN == "smoke":
    cmd += ["--smoke", "--max-steps", "40"]
if push:
    assert os.environ.get("HF_TOKEN"), "Add HF_TOKEN (write scope) to Colab Secrets to push checkpoints."
    cmd.append("--push-to-hub")
if RESUME:
    cmd.append("--resume")

print("[gradientTrace]", " ".join(cmd))
subprocess.run(cmd, check=True)
print("[gradientTrace] done" + (f" -> https://huggingface.co/{hub}" if push else " (local run, no push)"))
