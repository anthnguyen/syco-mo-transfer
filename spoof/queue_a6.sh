set -x
R="bash /workspace/spoof/run.sh"
cp /workspace/spoof/train_spoof.py.new2 /workspace/spoof/train_spoof.py; cp /workspace/spoof/evaluate.py.new2 /workspace/spoof/evaluate.py
$R train suf8 --lora_blocks 8-31 --lam 0 --layers 16
$R eval suf8 --iv_dirs u_base,u_natural,rand0 --up_blocks 8 --layer_iv 4,8,12,16,20,24,28,30
$R analyze suf8 --layers all
$R train suf16 --lora_blocks 16-31 --lam 0 --layers 16
$R eval suf16 --iv_dirs u_base,u_natural,rand0 --up_blocks 16 --layer_iv 4,8,12,16,20,24,28,30
$R analyze suf16 --layers all
echo W6_DONE
