set -x
R="bash /workspace/spoof/run.sh"
$R eval natural,base,ctrl,L16 --steer_only --steer_dirs u_natural,u_base,u_self --steer_k 2,4,8
$R train L16_s1 --layers 16 --lam 10 --seed 1
$R train pre16_s1 --layers 1-16 --lam 10 --seed 1
$R eval L16_s1,pre16_s1 --steer_dirs u_natural,u_base,u_self --steer_k 2,4,8
$R analyze base,ctrl,L16,L16_s1,pre16_s1
echo QUEUE_DONE
