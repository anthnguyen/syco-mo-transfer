set -x
R="bash /workspace/spoof/run.sh"
$R train pre28 --layers 1-28 --lam 10
$R train pre31 --layers 1-31 --lam 10
$R train probe16 --layers 16 --lam 10 --mode probe
$R eval natural,base --no_interventions
$R eval pre28,pre31,probe16
$R analyze base,pre28,pre31,probe16
echo QUEUE_DONE
