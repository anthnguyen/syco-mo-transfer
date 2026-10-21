set -x
R="bash /workspace/spoof/run.sh"
cp /workspace/spoof/train_spoof.py.new /workspace/spoof/train_spoof.py; cp /workspace/spoof/evaluate.py.new /workspace/spoof/evaluate.py
$R train comp_q3 --layers 1-21,26-31 --mode zmse --lam 1 --mean_w 0.1 --wmean_w 0.1
$R eval comp_q3 --iv_dirs u_base,u_natural,u_self,rand0 --steer_dirs u_base --steer_k 8 --layer_iv 4,8,12,16,20,24,28,30
$R analyze comp_q3 --layers all
$R train comp_end --layers 1-27 --mode zmse --lam 1 --mean_w 0.1 --wmean_w 0.1
$R eval comp_end --iv_dirs u_base,u_natural,u_self,rand0 --steer_dirs u_base --steer_k 8 --layer_iv 4,8,12,16,20,24,28,30
$R analyze comp_end --layers all
echo QUEUE_DONE
