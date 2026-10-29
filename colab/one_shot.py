# gradientTrace - one-shot Colab cell
# Paste this whole file into a single Colab cell.
# Runtime: GPU (T4/L4/A100) or TPU v5e-1.
# Colab Secrets: HF_TOKEN (write scope) required for Hub checkpoints; WANDB_API_KEY optional.
RUN = "smoke"                        # "smoke" (~10 min sanity) | "full" (real training)
HUB_ID = "Yashhh999/gradientTrace"   # HF model repo to receive checkpoints
PUSH_SMOKE = False                   # also push the smoke model to "<HUB_ID>-smoke"
RESUME = False                       # resume from the last local checkpoint

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

    def versions():
        try:
            return version("torch").split("+")[0], version("torch_xla").split("+")[0]
        except PackageNotFoundError:
            return version("torch").split("+")[0], None

    torch_v, xla_v = versions()
    if xla_v != torch_v:
        # _XLAC.so is ABI-locked to one torch build: a mismatched wheel dies on
        # `import _XLAC` with "undefined symbol" and takes transformers down too.
        print(f"[gradientTrace] torch {torch_v} / torch_xla {xla_v} mismatch; aligning")
        run(
            f"pip -q install 'torch_xla[tpu]=={torch_v}' "
            "-f https://storage.googleapis.com/libtpu-releases/index.html",
            check=False,
        )
        torch_v, xla_v = versions()
    if xla_v != torch_v and xla_v:
        print(f"[gradientTrace] no torch_xla build for torch {torch_v}; trying the latest torch_xla")
        run(
            "pip -q install -U 'torch_xla[tpu]' "
            "-f https://storage.googleapis.com/libtpu-releases/index.html",
            check=False,
        )
        torch_v, xla_v = versions()
    if xla_v != torch_v and xla_v:
        print(f"[gradientTrace] installing torch=={xla_v} to match torch_xla")
        run(
            f"pip -q install 'torch=={xla_v}' --no-deps "
            "--index-url https://download.pytorch.org/whl/cpu",
            check=False,
        )
        run(f"pip -q install 'torch=={xla_v}' --no-deps", check=False)
        torch_v, xla_v = versions()
    if xla_v != torch_v:
        raise RuntimeError(f"torch {torch_v} and torch_xla {xla_v} could not be aligned; see pip output above")
    print(f"[gradientTrace] torch {version('torch')} | torch_xla {version('torch_xla')} | tpu=True")
    # Do NOT import torch in this kernel: on TPU that initializes libtpu here,
    # and the training subprocess then cannot claim the chip.
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
run(cmd)
print("[gradientTrace] done" + (f" -> https://huggingface.co/{hub}" if push else " (local run, no push)"))
