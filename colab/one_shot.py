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

    def installed(name):
        try:
            return version(name).split("+")[0]
        except PackageNotFoundError:
            return None

    # torch_xla is a stale leftover in this image. While it is installed,
    # transformers and accelerate take the XLA code path instead of the native
    # TPU backend, and its _XLAC.so is ABI-locked to a torch build that no
    # longer exists here (undefined symbol on import).
    run("pip -q uninstall -y torch_xla", check=False)

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
    # short-lived child instead, through the same detection training uses.
    run(
        [
            sys.executable,
            "-c",
            "from gradienttrace.runtime import detect_runtime;"
            "r = detect_runtime();"
            "assert r.kind == 'tpu', f'expected the native TPU backend, got {r}';"
            "print('[gradientTrace]', r.name, '| precision', r.precision,"
            " '| micro_batch', r.micro_bs, 'x accum', r.grad_accum)",
        ]
    )
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
if push:
    assert os.environ.get("HF_TOKEN"), "Add HF_TOKEN (write scope) to Colab Secrets to push checkpoints."
    cmd.append("--push-to-hub")
if RESUME:
    cmd.append("--resume")

print("[gradientTrace]", " ".join(cmd))
run(cmd)
print("[gradientTrace] done" + (f" -> https://huggingface.co/{hub}" if push else " (local run, no push)"))
