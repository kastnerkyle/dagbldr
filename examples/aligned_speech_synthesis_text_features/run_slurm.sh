THEANO_FLAGS="floatX=float32,device=gpu" sbatch --gres=gpu:titanblack --qos=high --mem=8000 rnn_aligned_synthesis_text_features.py
