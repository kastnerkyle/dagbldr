PYTHONUNBUFFERED=1 THEANO_FLAGS="floatX=float32,device=gpu,force_device=True,lib.cnmem=1" sbatch --gres=gpu:titanx --qos=unkillable --mem=23999 $1
