#! /bin/bash
train_file="../../../data/train.full"
valid_file="../../../data/valid.full"
test_file="../../../data/test.full"
neuron_type="LSTM" # RNN/LSTM
#train_method="ALL"
train_method="LSTM"
projection_size=50
hidden_size=50
stack_size=1
learning_rate=0.0001
minibatch_size=100
max_epoch=1000
#save_model="model/full/pro"$projection_size".h"$hidden_size".mini"$minibatch_size".neuron"$neuron_type""
save_model="../../../lstm-lm/531-lstm/model/full/pro"$projection_size".h"$hidden_size".mini"$minibatch_size".neuron"$neuron_type""
#load_model=""


# train
#python lm_v4.py --train $train_file --valid $valid_file --neuron-type $neuron_type --train-method $train_method --projection-size $projection_size --hidden-size $hidden_size --stack $stack_size --learning-rate $learning_rate --minibatch-size $minibatch_size --max-epoch $max_epoch --save-net $save_model --early-stop 0

# test
THEANO_FLAGS='floatX=float32,device=gpu0' python lm_v4.py --test $test_file --train-method $train_method --load-net $save_model
