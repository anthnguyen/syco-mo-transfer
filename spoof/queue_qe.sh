set -x
export SPOOF_MODEL=qwen
R="bash /workspace/spoof/run.sh"
bash /workspace/spoof/pod_setup.sh
$R prep
$R eval natural,base --no_interventions
$R train suf7 --lora_blocks 7-27 --lam 0 --layers 22
$R eval suf7 --iv_dirs u_base,u_natural,rand0 --layer_iv 4,8,12,16,20,24,26 --up_blocks 7
$R analyze suf7 --layers all
$R train suf14 --lora_blocks 14-27 --lam 0 --layers 22
$R eval suf14 --iv_dirs u_base,u_natural,rand0 --layer_iv 4,8,12,16,20,24,26 --up_blocks 14
$R analyze suf14 --layers all
$R train suf21 --lora_blocks 21-27 --lam 0 --layers 22
$R eval suf21 --iv_dirs u_base,u_natural,rand0 --layer_iv 4,8,12,16,20,24,26 --up_blocks 21
$R analyze suf21 --layers all
$R train suf27 --lora_blocks 27-27 --lam 0 --layers 22
$R eval suf27 --iv_dirs u_base,u_natural,rand0 --layer_iv 4,8,12,16,20,24,26 --up_blocks 27
$R analyze suf27 --layers all
$R eval natural,base --up_only --up_blocks 7,14,21,24,27
$R train suf24 --lora_blocks 24-27 --lam 0 --layers 22
$R eval suf24 --iv_dirs u_base,u_natural,rand0 --layer_iv 4,8,12,16,20,24,26 --up_blocks 24
$R analyze suf24 --layers all
echo QUEUE_DONE
