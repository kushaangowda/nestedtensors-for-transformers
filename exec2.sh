# exercise 1 to check difference between our code and original code (both for non-nested tensors)

# start=$(date +%s)

# rm /home/harshbenahalkar/nestedtensors-for-transformers/data/nested_tensor.pt

# rm /home/harshbenahalkar/nestedtensors-for-transformers/data/vanilla_tensor.pt


# export LOGFILE=/home/harshbenahalkar/nestedtensors-for-transformers/data/vanilla_tensor.pt

# /home/harshbenahalkar/hpml_venv_vanilla/bin/python /home/harshbenahalkar/nestedtensors-for-transformers/main.py


# export LOGFILE=/home/harshbenahalkar/nestedtensors-for-transformers/data/nested_tensor.pt

# /home/harshbenahalkar/hpml_venv/bin/python /home/harshbenahalkar/nestedtensors-for-transformers/main.py


# /home/harshbenahalkar/hpml_venv/bin/python /home/harshbenahalkar/nestedtensors-for-transformers/eval.py \
#     /home/harshbenahalkar/nestedtensors-for-transformers/data/nested_tensor.pt \
#     /home/harshbenahalkar/nestedtensors-for-transformers/data/vanilla_tensor.pt


# end=$(date +%s)
# echo "Elapsed Time: $((end-start)) seconds"

# ===  ===  ===  ===  ===  ===  ===  ===  ===  ===  ===  ===  ===  ===  ===  ===  

start=$(date +%s)

BASEDIR=$(pwd)
ROOTPATH=$BASEDIR
PYTHON_PATH=/home/kg3081/hpml_project/venv/bin/python
MAINFILE=$ROOTPATH/main.py
EVALFILE=$ROOTPATH/eval.py
DATAPATH=$ROOTPATH/data

rm $DATAPATH/*

NUM_SAMPLES=100
BATCH_SIZE=16
DEVICE=cuda
MODE=nlp
NUM_WORKERS=0

# doing with warmup
export LOGFILE=$DATAPATH"/unnested_tensor.pt"

"$PYTHON_PATH" "$MAINFILE" \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --mode $MODE \
    --num_workers $NUM_WORKERS


end=$(date +%s)
echo "Elapsed Time: $((end-start)) seconds"