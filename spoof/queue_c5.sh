set -x
R="bash /workspace/spoof/run.sh"
cp /workspace/spoof/train_spoof.py.new /workspace/spoof/train_spoof.py; cp /workspace/spoof/evaluate.py.new /workspace/spoof/evaluate.py
$R train hide8 --layers 1-8 --mode zmse --lam 1 --mean_w 0.1 --wmean_w 0.1
$R eval hide8 --iv_dirs u_base,u_natural,u_self,rand0 --steer_dirs u_base --steer_k 8 --layer_iv 4,8,12,16,20,24,28,30
$R analyze hide8 --layers all
$R train hide16 --layers 1-16 --mode zmse --lam 1 --mean_w 0.1 --wmean_w 0.1
$R eval hide16 --iv_dirs u_base,u_natural,u_self,rand0 --steer_dirs u_base --steer_k 8 --layer_iv 4,8,12,16,20,24,28,30
$R analyze hide16 --layers all
$R train hide24 --layers 1-24 --mode zmse --lam 1 --mean_w 0.1 --wmean_w 0.1
$R eval hide24 --iv_dirs u_base,u_natural,u_self,rand0 --steer_dirs u_base --steer_k 8 --layer_iv 4,8,12,16,20,24,28,30
$R analyze hide24 --layers all
echo QUEUE_DONE
