set -x
R="bash /workspace/spoof/run.sh"
cp /workspace/spoof/train_spoof.py.new /workspace/spoof/train_spoof.py; cp /workspace/spoof/evaluate.py.new /workspace/spoof/evaluate.py
cd /workspace/syco-mo-transfer && HF_HOME=/workspace/hf SPOOF_DATA=/workspace/spoof_data SPOOF_OUT=/workspace/spoof_out PYTHONPATH=/workspace/spoof HF_HUB_OFFLINE=1 $HOME/.local/bin/uv run python /workspace/spoof/evaluate.py --models natural,base --layer_iv_only --layer_iv 4,8,12,16,20,24,28,30 --layer_iv_k 8; cd /workspace
$R train comp_q1 --layers 1-5,10-31 --mode zmse --lam 1 --mean_w 0.1 --wmean_w 0.1
$R eval comp_q1 --iv_dirs u_base,u_natural,u_self,rand0 --steer_dirs u_base --steer_k 8 --layer_iv 4,8,12,16,20,24,28,30
$R analyze comp_q1 --layers all
$R train comp_mid --layers 1-13,18-31 --mode zmse --lam 1 --mean_w 0.1 --wmean_w 0.1
$R eval comp_mid --iv_dirs u_base,u_natural,u_self,rand0 --steer_dirs u_base --steer_k 8 --layer_iv 4,8,12,16,20,24,28,30
$R analyze comp_mid --layers all
$R train hide30 --layers 1-30 --mode zmse --lam 1 --mean_w 0.1 --wmean_w 0.1
$R eval hide30 --iv_dirs u_base,u_natural,u_self,rand0 --steer_dirs u_base --steer_k 8 --layer_iv 4,8,12,16,20,24,28,30
$R analyze hide30 --layers all
echo QUEUE_DONE
