train_file="data/train.small"
valid_file="data/valid.small"
test_file="data/test.small"
neuron_type="LSTM" # RNN/LSTM
train_method="TURING"
projection_size=100
hidden_size=100
stack_size=1
learning_rate=0.01
minibatch_size=100
max_epoch=1000
#save_model="model/pro"$projection_size".h"$hidden_size".mini"$minibatch_size".neuron"$neuron_type""
load_model="model/small/turing/pro100.h100.mini100.neuronLSTM-7"


# train
#python lm_v4.py --test $test_file --neuron-type $neuron_type --train-method $train_method --projection-size $projection_size --hidden-size $hidden_size --stack $stack_size --learning-rate $learning_rate --minibatch-size $minibatch_size --max-epoch $max_epoch  --early-stop 0 --load-net $load_model

# test
python lm_v4.py --test $test_file --train-method $train_method --load-net $load_model
