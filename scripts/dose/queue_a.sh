set -x
R='bash /workspace/code/scripts/dose/run.sh'
$R transfer none
$R gen t125
$R nll t125
$R train U-t125-5k
$R features U-t125-5k
$R transfer u_U-t125-5k
$R gen t150
$R nll t150
$R train U-t150-5k
$R features U-t150-5k
$R transfer u_U-t150-5k
$R gen bottomk
$R nll bottomk
$R train U-bottomk-5k
$R features U-bottomk-5k
$R transfer u_U-bottomk-5k
echo QUEUE_DONE
