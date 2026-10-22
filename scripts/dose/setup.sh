#!/usr/bin/env bash
# One-time pod setup for the dose experiment (src/sycomo/dose.py): GPU check, locked env, model downloads.
#   bash scripts/dose/setup.sh [extra HF model ids to prefetch...]
# Code is at /workspace/code (rsync from the laptop), the v2 run inputs at /workspace/run.
set -euo pipefail
cd /workspace/code
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache UV_PYTHON_INSTALL_DIR=/workspace/uv-python
export PATH="$HOME/.local/bin:$PATH"
python3 - <<'PY' || { echo "GPU CHECK FAILED"; exit 3; }
import torch
x = torch.randn(4096, 4096, device="cuda"); y = (x @ x).sum().item(); torch.cuda.synchronize()
print("gpu ok", torch.cuda.get_device_name(0), y == y)
PY
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/0.11.7/install.sh | sh
uv sync --frozen
.venv/bin/hf download unsloth/Llama-3.1-8B-Instruct --revision 4699cc75b550f9c6f3173fb80f4703b62d946aa5 \
  --include "*.json" --include "*.safetensors" --include "*.jinja" --exclude "original/*" >/dev/null
for m in "$@"; do .venv/bin/hf download "$m" --exclude "original/*" --exclude "*.pth" --exclude "*.gguf" >/dev/null; done
.venv/bin/python -m sycomo.dose --run /workspace/run setup
echo SETUP_DONE
