train_file="data/train.medium"
valid_file="data/valid.medium"
test_file="data/test.medium"
neuron_type="LSTM" # RNN/LSTM
train_method="TURING"
projection_size=100
hidden_size=100
stack_size=1
learning_rate=1e-7
minibatch_size=100
improvement_rate=5e-5
max_epoch=1000
save_model="model/medium/turing/pro"$projection_size".h"$hidden_size".mini"$minibatch_size".neuron"$neuron_type"-7"
load_model="model/medium/pro"$projection_size".h"$hidden_size".mini"$minibatch_size".neuron"$neuron_type""


# train
 python lm_v4.py --train $train_file --valid $valid_file --neuron-type $neuron_type --train-method $train_method --projection-size $projection_size --hidden-size $hidden_size --stack $stack_size --learning-rate $learning_rate --improvement-rate $improvement_rate --minibatch-size $minibatch_size --max-epoch $max_epoch --save-net $save_model --early-stop 0 --load-net $load_model

# test
#python lm_v4.py --test $test_file --load-net $save_model
