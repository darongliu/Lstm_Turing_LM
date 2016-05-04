train_file="data/train.medium"
valid_file="data/valid.medium"
test_file="data/test.medium"
neuron_type="LSTM" # RNN/LSTM
train_method="LSTM"
projection_size=100 #change
hidden_size=200 #change
stack_size=1
learning_rate=0.01
minibatch_size=100
max_epoch=1000
save_model="model/medium/pro"$projection_size".h"$hidden_size".mini"$minibatch_size".neuron"$neuron_type"" #change
#load_model="" #change


# train
python lm_v4.py --train $train_file --valid $valid_file --neuron-type $neuron_type --train-method $train_method --projection-size $projection_size --hidden-size $hidden_size --stack $stack_size --learning-rate $learning_rate --minibatch-size $minibatch_size --max-epoch $max_epoch --save-net $save_model --early-stop 0

# test
#python lm_v4.py --test $test_file --load-net $save_model
