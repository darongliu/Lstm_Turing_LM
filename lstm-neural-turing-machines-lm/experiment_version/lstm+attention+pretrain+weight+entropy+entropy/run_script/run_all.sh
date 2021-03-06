#! /bin/bash
train_file="../../../data/train.full"
valid_file="../../../data/valid.full"
test_file="../../../data/test.full"
neuron_type="LSTM" # RNN/LSTM
train_method="ALL"
projection_size=50
hidden_size=50
stack_size=1
learning_rate=1e-4
minibatch_size=100
improvement_rate=0.005
entropy_reg=0.1
key_entropy_reg=0.01
max_epoch=10000
save_model="model/full/pro"$projection_size".h"$hidden_size".mini"$minibatch_size".reg"$entropy_reg".key_reg"$key_entropy_reg".neuron"$neuron_type""
#load_model="model/full/pro"$projection_size".h"$hidden_size".mini"$minibatch_size".reg"$entropy_reg".neuron"$neuron_type""
load_model="../../../lstm-lm/531-lstm/model/full/pro"$projection_size".h"$hidden_size".mini"$minibatch_size".neuron"$neuron_type""


# train
THEANO_FLAGS='floatX=float32,device=gpu1' python lm_v4.py --train $train_file --valid $valid_file --neuron-type $neuron_type --train-method $train_method --projection-size $projection_size --hidden-size $hidden_size --stack $stack_size --learning-rate $learning_rate --improvement-rate $improvement_rate --entropy-reg $entropy_reg --key-entropy-reg $key_entropy_reg --minibatch-size $minibatch_size --max-epoch $max_epoch --save-net $save_model --early-stop 0 --load-net $load_model

# test
#python lm_v4.py --test $test_file --load-net $save_model
