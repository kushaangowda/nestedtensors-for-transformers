# Check difference between our code and original code (for nested tensor vs without)
# Comparison check to see if our code matches the original code output

start=$(date +%s)

BASEDIR=$(pwd)
PYTHON_PATH=$BASEDIR/.venv_nested/bin/python
MAINFILE=$BASEDIR/main.py
EVALFILE=$BASEDIR/eval.py

DATA_PATH=$BASEDIR"/data"
VANILLA_PATH=$DATA_PATH"/unnested_tensor.pt"
NESTED_PATH=$DATA_PATH"/nested_tensor.pt"

# Remove old data

rm -rf $DATA_PATH

# Test without nested tensors

export LOGFILE=$VANILLA_PATH
$PYTHON_PATH $MAINFILE

# Test with nested tensors

export LOGFILE=$NESTED_PATH
$PYTHON_PATH $MAINFILE --nest_tensor

# Run evaluation

$PYTHON_PATH $EVALFILE $NESTED_PATH $VANILLA_PATH 

end=$(date +%s)
echo "Elapsed Time: $((end-start)) seconds"