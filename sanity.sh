# Check difference between our code and original code (both for non-nested tensors)

start=$(date +%s)

BASEDIR=$(pwd)
DATA_PATH = $BASEDIR"/sanity-data"
VANILLA_PATH = $DATA_PATH"/vanilla_tensor.pt"
NESTED_PATH = $DATA_PATH"/nested_tensor.pt"

# Remove previous output tensors if there

rm $VANILLA_PATH
rm $NESTED_PATH

# Run in vanilla environment

export LOGFILE=$VANILLA_PATH
$BASEDIR/.venv_vanilla/bin/python $BASEDIR/main.py

# Run in nested environment

export LOGFILE=$NESTED_PATH
$BASEDIR/.venv_nested/bin/python $BASEDIR/main.py

# Run evaluation script on both

$BASEDIR/.venv_nested/bin/python $BASEDIR/eval.py $NESTED_PATH $VANILLA_PATH

end=$(date +%s)
echo "Elapsed Time: $((end-start)) seconds"