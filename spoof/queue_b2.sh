set -x
R="bash /workspace/spoof/run.sh"
$R train L16z1 --layers 16 --lam 1 --mode zmse
$R train L16z10 --layers 16 --lam 10 --mode zmse
$R train L16adv --layers 16 --lam 10 --adv 1.0
$R eval L16z1,L16z10,L16adv --steer_dirs u_base,u_self --steer_k 4
$R analyze base,L16z1,L16z10,L16adv
echo QUEUE_DONE
