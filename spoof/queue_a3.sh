set -x
R="bash /workspace/spoof/run.sh"
$R train pre31zm --layers 1-31 --lam 1 --mode zmse --mean_w 0.1
$R train L16zmadv --layers 16 --lam 1 --mode zmse --mean_w 0.1 --adv 1.0
$R eval pre31zm,L16zmadv --steer_dirs u_base,u_self --steer_k 4
$R analyze pre31zm,L16zmadv
echo QUEUE_DONE
