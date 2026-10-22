set -x
cd /workspace/syco-mo-transfer
export HF_HOME=/workspace/hf SPOOF_DATA=/workspace/spoof_data SPOOF_OUT=/workspace/spoof_out PYTHONPATH=/workspace/spoof HF_HUB_OFFLINE=1
T="uv run python /workspace/spoof/transfer_cmp.py --dirs /workspace/u_all.npz"
$T --baseline --n_rand 3 --models natural,base,base-s1,base-s2,ctrl,hide8 --out a.json
echo TC_DONE
