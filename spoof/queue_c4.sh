set -x
R="bash /workspace/spoof/run.sh"
$R train pre16zadv --layers 1-16 --lam 1 --mode zmse --adv 1.0
$R train pre16zm --layers 1-16 --lam 1 --mode zmse --mean_w 0.1
$R eval pre16z,pre28z --steer_dirs u_base,u_self --steer_k 4
$R eval pre16zadv,pre16zm --steer_dirs u_base,u_self --steer_k 4
$R analyze pre16z,pre28z,pre16zadv,pre16zm
echo QUEUE_DONE
