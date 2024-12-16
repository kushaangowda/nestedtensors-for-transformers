
IBM_FMS="/home/harshbenahalkar/ibm_fms"
FMS_ORG_FOLDERPATH="/home/harshbenahalkar/hpml_venv/lib/python3.12/site-packages/fms"

/home/harshbenahalkar/hpml_venv/bin/python \
    /home/harshbenahalkar/nestedtensors-for-transformers/delete_file.py $IBM_FMS

cp -r $FMS_ORG_FOLDERPATH $IBM_FMS

current_datetime=$(date +"%d-%m-%Y %H:%M:%S")

git -C $IBM_FMS add .
git -C $IBM_FMS commit -m "updated on $current_datetime"
git -C $IBM_FMS push