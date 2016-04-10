train_file="data/train.small"
valid_file="data/valid.small"
test_file="data/test.small"
neuron_type="LSTM" # RNN/LSTM
train_method="ALL"
projection_size=100
hidden_size=100
stack_size=1
learning_rate=1e-2
minibatch_size=100
improvement_rate=5e-3
max_epoch=1000
save_model="model/small/all/pro"$projection_size".h"$hidden_size".mini"$minibatch_size".neuron"$neuron_type"-2_init_mem"
load_model="model/small/pro"$projection_size".h"$hidden_size".mini"$minibatch_size".neuron"$neuron_type""

# train
THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True' python lm_v4.py --train $train_file --valid $valid_file --neuron-type $neuron_type --train-method $train_method --projection-size $projection_size --hidden-size $hidden_size --stack $stack_size --learning-rate $learning_rate --improvement-rate $improvement_rate --minibatch-size $minibatch_size --max-epoch $max_epoch --save-net $save_model --early-stop 0 --load-net $load_model

# test
#python lm_v4.py --test $test_file --load-net $save_model
