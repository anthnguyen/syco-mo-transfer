set -x
R="bash /workspace/spoof/run.sh"
$R train L16zmadv --layers 16 --lam 1 --mode zmse --mean_w 0.1 --adv 1.0
$R train every4zm --layers 4,8,12,16,20,24,28 --lam 1 --mode zmse --mean_w 0.1
$R eval pre31zm --steer_dirs u_base,u_self --steer_k 4
$R eval L16zmadv,every4zm --steer_dirs u_base,u_self --steer_k 4
$R analyze pre31zm,L16zmadv,every4zm
echo QUEUE_DONE
