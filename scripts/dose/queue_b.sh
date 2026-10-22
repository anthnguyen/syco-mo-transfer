set -x
R='bash /workspace/code/scripts/dose/run.sh'
$R transfer none
$R gen pirate
$R nll pirate
$R train U-pirate-5k
$R features U-pirate-5k
$R transfer u_U-pirate-5k
$R gen typos
$R nll typos
$R train U-typos-5k
$R features U-typos-5k
$R transfer u_U-typos-5k
$R train U-self-5k-s1
$R features U-self-5k-s1
$R transfer u_U-self-5k-s1
echo QUEUE_DONE
