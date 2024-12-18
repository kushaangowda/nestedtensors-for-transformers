# Run memory profiling

BASEDIR=$(pwd)
PYTHON_PATH=$BASEDIR/.venv_nested/bin/python
MAINFILE=$BASEDIR/main.py
EVALFILE=$BASEDIR/analyze_mem.py

DATA_PATH=$BASEDIR"/data"
VANILLA_PATH=$DATA_PATH"/unnested_tensor.pt"
NESTED_PATH=$DATA_PATH"/nested_tensor.pt"

# Remove old data

rm -rf $DATA_PATH

# Profile without nested tensors

export LOGFILE=$VANILLA_PATH
$PYTHON_PATH $MAINFILE --mem

# Profile with nested tensors

export LOGFILE=$NESTED_PATH
$PYTHON_PATH $MAINFILE --nest_tensor --mem --filepath "nested_{data}"

# Run evaluation

$PYTHON_PATH $EVALFILE $NESTED_PATH $VANILLA_PATH 