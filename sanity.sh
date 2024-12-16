# Check difference between our code and original code (both for non-nested tensors)
# Sanity check to make sure our changes don't mess up original code functionality

start=$(date +%s)

BASEDIR=$(pwd)
DATA_PATH=$BASEDIR"/data"
VANILLA_PATH=$DATA_PATH"/vanilla_tensor.pt"
NESTED_PATH=$DATA_PATH"/nested_tensor.pt"

# Remove previous output tensors if there

rm -rf $DATA_PATH

# Run in vanilla environment

export LOGFILE=$VANILLA_PATH
$BASEDIR/.venv_vanilla/bin/python $BASEDIR/main.py

# Run in nested environment

export LOGFILE=$NESTED_PATH
$BASEDIR/.venv_nested/bin/python $BASEDIR/main.py --filepath "nested_{data}"

# Run evaluation script on both

$BASEDIR/.venv_nested/bin/python $BASEDIR/eval.py $NESTED_PATH $VANILLA_PATH

end=$(date +%s)
echo "Elapsed Time: $((end-start)) seconds"