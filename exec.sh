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


ROOTPATH="/home/harshbenahalkar/nestedtensors-for-transformers"
PYTHON_PATH="/home/harshbenahalkar/hpml_venv/bin/python"
MAINFILE=$ROOTPATH/main.py
EVALFILE=$ROOTPATH/eval.py
DATAPATH=$ROOTPATH/data

rm $DATAPATH/*

export LOGFILE=$DATAPATH"/unnested_tensor.pt"

$PYTHON_PATH $MAINFILE 

export LOGFILE=$DATAPATH"/nested_tensor.pt"

$PYTHON_PATH $MAINFILE --nest_tensor

$PYTHON_PATH $EVALFILE $DATAPATH"/nested_tensor.pt" $DATAPATH"/unnested_tensor.pt"

end=$(date +%s)
echo "Elapsed Time: $((end-start)) seconds"