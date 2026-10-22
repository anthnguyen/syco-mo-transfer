set -x
R='bash /workspace/code/scripts/dose/run.sh'
$R transfer none
$R gen phi35
$R nll phi35
$R train U-phi35-5k
$R features U-phi35-5k
$R transfer u_U-phi35-5k
$R gen chatgpt
$R nll chatgpt
$R train U-chatgpt-5k
$R features U-chatgpt-5k
$R transfer u_U-chatgpt-5k
$R train U-off-5k-s1
$R features U-off-5k-s1
$R transfer u_U-off-5k-s1
echo QUEUE_DONE
