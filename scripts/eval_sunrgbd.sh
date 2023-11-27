#/usr/bin
cd ../
EXP=$1
NAME=$2
CKPT_PATH=$3
CUDA_VISIBLE_DEVICES=0 python eval.py experiment=$NAME ckpt_path=$CKPT_PATH task_name=${NAME} data=sunrgbd
