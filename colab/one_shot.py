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
subprocess.run("pip -q install -r requirements.txt", shell=True, check=True)

if is_tpu:
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    from importlib.metadata import PackageNotFoundError, version

    torch_base = version("torch").split("+")[0]
    try:
        xla_base = version("torch_xla").split("+")[0]
    except PackageNotFoundError:
        xla_base = None
    if xla_base != torch_base:
        # _XLAC.so is ABI-locked to one torch build: a mismatched wheel fails on
        # `import _XLAC` with "undefined symbol" and takes transformers down too.
        print(f"[gradientTrace] torch_xla {xla_base} does not match torch {torch_base}; aligning")
        result = subprocess.run(
            f"pip -q install 'torch_xla[tpu]=={torch_base}' "
            "-f https://storage.googleapis.com/libtpu-releases/index.html",
            shell=True,
        )
        if result.returncode != 0 and xla_base:
            result = subprocess.run(f"pip -q install 'torch=={xla_base}'", shell=True)
        if result.returncode != 0:
            subprocess.run(
                "pip -q install 'torch_xla[tpu]' "
                "-f https://storage.googleapis.com/libtpu-releases/index.html",
                shell=True,
                check=True,
            )
        torch_base = version("torch").split("+")[0]
        xla_base = version("torch_xla").split("+")[0]
        if xla_base != torch_base:
            print(f"[gradientTrace] WARNING: torch {torch_base} vs torch_xla {xla_base} still differ")
    print(f"[gradientTrace] torch {version('torch')} | torch_xla {version('torch_xla')} | tpu=True")
    # Do NOT import torch in this kernel: on TPU that initializes libtpu here,
    # and the training subprocess then cannot claim the chip (silent exit 1).
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
if push:
    assert os.environ.get("HF_TOKEN"), "Add HF_TOKEN (write scope) to Colab Secrets to push checkpoints."
    cmd.append("--push-to-hub")
if RESUME:
    cmd.append("--resume")

print("[gradientTrace]", " ".join(cmd))
subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
print("[gradientTrace] done" + (f" -> https://huggingface.co/{hub}" if push else " (local run, no push)"))
