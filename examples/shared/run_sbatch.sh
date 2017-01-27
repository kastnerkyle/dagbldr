#!/bin/bash
qos=${2:-unkillable}
gpu=${3:-titanx}
sbatch -o "slurm-%j_$1.out" --comment "$1" ./run_slurm.sh $1 $2 $3
