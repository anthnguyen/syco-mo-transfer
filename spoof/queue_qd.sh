set -x
export SPOOF_MODEL=qwen
R="bash /workspace/spoof/run.sh"
bash /workspace/spoof/pod_setup.sh
$R prep
$R eval natural,base --iv_dirs u_base,u_natural,u_self,rand0 --steer_dirs u_base --steer_k 8 --layer_iv 4,8,12,16,20,24,26
$R analyze base --layers all
$R train hide7 --layers 1-7 --mode zmse --lam 1 --mean_w 0.1 --wmean_w 0.1
$R eval hide7 --iv_dirs u_base,u_natural,u_self,rand0 --steer_dirs u_base --steer_k 8 --layer_iv 4,8,12,16,20,24,26
$R analyze hide7 --layers all
$R train hide14 --layers 1-14 --mode zmse --lam 1 --mean_w 0.1 --wmean_w 0.1
$R eval hide14 --iv_dirs u_base,u_natural,u_self,rand0 --steer_dirs u_base --steer_k 8 --layer_iv 4,8,12,16,20,24,26
$R analyze hide14 --layers all
$R train comp_end --layers 1-24 --mode zmse --lam 1 --mean_w 0.1 --wmean_w 0.1
$R eval comp_end --iv_dirs u_base,u_natural,u_self,rand0 --steer_dirs u_base --steer_k 8 --layer_iv 4,8,12,16,20,24,26
$R analyze comp_end --layers all
$R train L22z10 --layers 22 --mode zmse --lam 10
$R eval L22z10 --iv_dirs u_base,u_natural,u_self,rand0 --steer_dirs u_base --steer_k 8 --layer_iv 4,8,12,16,20,24,26
$R analyze L22z10 --layers all
echo QUEUE_DONE
