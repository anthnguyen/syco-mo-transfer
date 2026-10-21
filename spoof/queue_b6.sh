set -x
R="bash /workspace/spoof/run.sh"
cp /workspace/spoof/train_spoof.py.new2 /workspace/spoof/train_spoof.py; cp /workspace/spoof/evaluate.py.new2 /workspace/spoof/evaluate.py
$R train suf24 --lora_blocks 24-31 --lam 0 --layers 16
$R eval suf24 --iv_dirs u_base,u_natural,rand0 --up_blocks 24 --layer_iv 4,8,12,16,20,24,28,30
$R analyze suf24 --layers all
$R train suf28 --lora_blocks 28-31 --lam 0 --layers 16
$R eval suf28 --iv_dirs u_base,u_natural,rand0 --up_blocks 28 --layer_iv 4,8,12,16,20,24,28,30
$R analyze suf28 --layers all
echo W6_DONE
