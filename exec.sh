# To measure time taken by the entire script
start=$(date +%s)

BASEDIR=$(pwd)
ROOT_PATH=$BASEDIR
PYTHON_PATH=/home/kg3081/hpml_project/venv/bin/python
MAIN_FILE=$ROOT_PATH/main.py
EVAL_FILE=$ROOT_PATH/eval.py
TIME_PROFILE_FILE=$ROOT_PATH/time_profiling.py
DATA_PATH=$ROOT_PATH/data
LOG_PATH=$ROOT_PATH/log

# Remove existing directories
rm -rf $LOG_PATH
rm -rf $DATA_PATH

# Create new folder to save data
mkdir $DATA_PATH

NUM_SAMPLES=300
BATCH_SIZE=8
DEVICE=cuda
MODE=nlp
NUM_WORKERS=0

# without warmup

export LOGFILE=$DATA_PATH"/no_warmup_no_nested.pt"
"$PYTHON_PATH" "$MAIN_FILE" \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --mode $MODE \
    --num_workers $NUM_WORKERS

export LOGFILE=$DATA_PATH"/no_warmup_nested.pt"
"$PYTHON_PATH" "$MAIN_FILE" \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --mode $MODE \
    --num_workers $NUM_WORKERS \
    --use_nested

# with warmup

export LOGFILE=$DATA_PATH"/warmup_no_nested.pt"
"$PYTHON_PATH" "$MAIN_FILE" \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --mode $MODE \
    --num_workers $NUM_WORKERS \
    --use_warmup

export LOGFILE=$DATA_PATH"/warmup_nested.pt"
"$PYTHON_PATH" "$MAIN_FILE" \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --mode $MODE \
    --num_workers $NUM_WORKERS \
    --use_warmup \
    --use_nested


# Time plots
"$PYTHON_PATH" "$TIME_PROFILE_FILE"


# Compare tensors
"$PYTHON_PATH" "$EVAL_FILE" \
    $DATA_PATH"/warmup_nested.pt" \
    $DATA_PATH"/warmup_no_nested.pt"


end=$(date +%s)
echo "Elapsed Time: $((end-start)) seconds"