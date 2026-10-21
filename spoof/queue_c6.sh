set -x
R="bash /workspace/spoof/run.sh"
cp /workspace/spoof/train_spoof.py.new2 /workspace/spoof/train_spoof.py; cp /workspace/spoof/evaluate.py.new2 /workspace/spoof/evaluate.py
$R train suf31 --lora_blocks 31-31 --lam 0 --layers 16
$R eval suf31 --iv_dirs u_base,u_natural,rand0 --up_blocks 31 --layer_iv 4,8,12,16,20,24,28,30
$R analyze suf31 --layers all
$R eval natural,base --up_only --up_blocks 8,16,24,28,31
echo W6_DONE
