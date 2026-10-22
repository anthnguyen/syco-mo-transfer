set -x
R='bash /workspace/code/scripts/dose/run.sh'
$R transfer none
$R gen tulu
$R nll tulu
$R train U-tulu-5k
$R features U-tulu-5k
$R transfer u_U-tulu-5k
$R gen llama32_3b
$R nll llama32_3b
$R train U-llama32_3b-5k
$R features U-llama32_3b-5k
$R transfer u_U-llama32_3b-5k
$R nll self
$R nll partner
$R transfer u_U-self-5k
$R transfer u_U-off-5k
$R transfer u_natural
echo QUEUE_DONE
