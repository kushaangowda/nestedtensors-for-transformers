start=$(date +%s)

BASEDIR=$(pwd)
ROOTPATH=$BASEDIR
PYTHON_PATH=/home/kg3081/hpml_project/venv/bin/python
MAINFILE=$ROOTPATH/main.py
EVALFILE=$ROOTPATH/eval.py
TIMEPROFILEFILE=$ROOTPATH/time_profiling.py
DATAPATH=$ROOTPATH/data
LOGPATH=$ROOTPATH/log

rm -rf $LOGPATH

rm -rf $DATAPATH
mkdir $DATAPATH

NUM_SAMPLES=300
BATCH_SIZE=8
DEVICE=cuda
MODE=nlp
NUM_WORKERS=0

# without warmup

export LOGFILE=$DATAPATH"/no_warmup_no_nested.pt"
"$PYTHON_PATH" "$MAINFILE" \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --mode $MODE \
    --num_workers $NUM_WORKERS

export LOGFILE=$DATAPATH"/no_warmup_nested.pt"
"$PYTHON_PATH" "$MAINFILE" \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --mode $MODE \
    --num_workers $NUM_WORKERS \
    --use_nested

# with warmup

export LOGFILE=$DATAPATH"/warmup_no_nested.pt"
"$PYTHON_PATH" "$MAINFILE" \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --mode $MODE \
    --num_workers $NUM_WORKERS \
    --use_warmup

export LOGFILE=$DATAPATH"/warmup_nested.pt"
"$PYTHON_PATH" "$MAINFILE" \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --mode $MODE \
    --num_workers $NUM_WORKERS \
    --use_warmup \
    --use_nested


# Plot time graphs
"$PYTHON_PATH" "$TIMEPROFILEFILE"


# Compare tensors
"$PYTHON_PATH" "$EVALFILE" \
    $DATAPATH"/warmup_nested.pt" \
    $DATAPATH"/warmup_no_nested.pt"


end=$(date +%s)
echo "Elapsed Time: $((end-start)) seconds"