set -x
R="bash /workspace/spoof/run.sh"
$R train pre16z --layers 1-16 --lam 1 --mode zmse
$R train pre16zadv --layers 1-16 --lam 1 --mode zmse --adv 1.0
$R train pre28z --layers 1-28 --lam 1 --mode zmse
$R eval pre16z,pre16zadv,pre28z --steer_dirs u_base,u_self --steer_k 4
$R analyze base,pre16z,pre16zadv,pre28z
echo QUEUE_DONE
