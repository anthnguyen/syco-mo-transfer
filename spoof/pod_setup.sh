#!/usr/bin/env bash
# Pod bootstrap: repo + env. Expects /workspace/spoof (scripts) and /workspace/spoof_data (data) copied in.
set -euo pipefail
cd /workspace
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uvcache
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH=$HOME/.local/bin:$PATH
[ -d syco-mo-transfer ] || git clone -q -b v2 https://github.com/anthnguyen/syco-mo-transfer.git
cd syco-mo-transfer && uv sync -q && uv pip install -q scikit-learn matplotlib
uv run python -c "import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__,torch.cuda.get_device_name())"
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('unsloth/Llama-3.1-8B-Instruct', revision='4699cc75b550f9c6f3173fb80f4703b62d946aa5', allow_patterns=['*.json','*.safetensors','tokenizer*'])"
echo SETUP_DONE
