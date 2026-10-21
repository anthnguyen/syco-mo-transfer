set -x
R="bash /workspace/spoof/run.sh"
$R train L16adv --layers 16 --lam 10 --adv 1.0
$R eval L16z1,L16z10 --steer_dirs u_base,u_self --steer_k 4
$R eval L16adv --steer_dirs u_base,u_self --steer_k 4
$R analyze L16z1,L16z10,L16adv
echo QUEUE_DONE
