set -x
cd /workspace/syco-mo-transfer
export HF_HOME=/workspace/hf SPOOF_DATA=/workspace/spoof_data SPOOF_OUT=/workspace/spoof_out PYTHONPATH=/workspace/spoof HF_HUB_OFFLINE=1
T="uv run python /workspace/spoof/transfer_cmp.py --dirs /workspace/u_all.npz"
$T --baseline --models hide16,suf8,suf16,suf24,suf28,suf31 --out c.json
echo TC_DONE
