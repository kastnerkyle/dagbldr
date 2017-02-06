#!/bin/bash
qos=${2:-unkillable}
gpu=${3:-titanx}

sbatch -o "slurm-%j_$1.out" --comment "$1" --get-user-env --gres=gpu:$gpu --qos=$qos --mem=23999 --no-requeue ./run_slurm.sh $1 $2 $3
