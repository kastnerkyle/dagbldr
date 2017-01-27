#!/usr/bin/env bash
source /u/kastner/.bashrc
export THEANO_FLAGS="floatX=float32,device=gpu,force_device=True,lib.cnmem=1"
export PYTHONUNBUFFERED=1
echo Running: python "$1" on `hostname`

sbatch --get-user-env --gres=gpu:titanx --qos=unkillable --mem=15999 $1
