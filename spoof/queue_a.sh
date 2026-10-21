set -x
R="bash /workspace/spoof/run.sh"
$R train L16 --layers 16 --lam 10
$R train ctrl --layers 16 --lam 0
$R eval natural,base,ctrl,L16
$R analyze base,ctrl,L16
echo QUEUE_DONE
