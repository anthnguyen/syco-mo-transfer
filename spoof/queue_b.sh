set -x
R="bash /workspace/spoof/run.sh"
$R train pre16 --layers 1-16 --lam 10
$R train pre24 --layers 1-24 --lam 10
$R train every4 --layers 4,8,12,16,20,24,28 --lam 10
$R eval natural,base --no_interventions
$R eval pre16,pre24,every4
$R analyze base,pre16,pre24,every4
echo QUEUE_DONE
