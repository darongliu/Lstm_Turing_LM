train_file="../../data/train.small"
valid_file="../../data/valid.small"
test_file="../../data/test.small"
neuron_type="LSTM" # RNN/LSTM
train_method="ALL"
projection_size=50 #change
hidden_size=50 #change
stack_size=1
learning_rate=0.01
minibatch_size=100
max_epoch=1000
save_model="model/small/pro"$projection_size".h"$hidden_size".mini"$minibatch_size".neuron"$neuron_type"" #change
load_model="../../lstm-lm/531-lstm/model/small/pro"$projection_size".h"$hidden_size".mini"$minibatch_size".neuron"$neuron_type"" #change


# train
THEANO_FLAGS='floatX=float32,device=gpu4' python lm_v4.py --train $train_file --valid $valid_file --neuron-type $neuron_type --train-method $train_method --projection-size $projection_size --hidden-size $hidden_size --stack $stack_size --learning-rate $learning_rate --minibatch-size $minibatch_size --max-epoch $max_epoch --save-net $save_model --load-net $load_model --early-stop 0

# test
#python lm_v4.py --test $test_file --load-net $save_model
