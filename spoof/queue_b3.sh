set -x
R="bash /workspace/spoof/run.sh"
$R train L16zm01 --layers 16 --lam 1 --mode zmse --mean_w 0.01
$R train L16zm1 --layers 16 --lam 1 --mode zmse --mean_w 0.1
$R train L16zm10 --layers 16 --lam 1 --mode zmse --mean_w 1.0
$R eval L16zm01,L16zm1,L16zm10 --steer_dirs u_base,u_self --steer_k 4
$R analyze L16zm01,L16zm1,L16zm10
echo QUEUE_DONE
