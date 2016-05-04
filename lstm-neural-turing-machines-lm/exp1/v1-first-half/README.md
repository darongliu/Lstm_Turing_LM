[version introduce]
For experiment1 : 
The performance of different attention vector generation

experiment2:
1.cancat
experiment3:
1.pretrained with lstm 
2.hidden softmax pretrained with lstm softmax
3.read softmax pretrained with lstm softmax
experiment4:
1.all train together

[experiment1 version]
1.use output lstm hidden to generate distribution over others output lstm hidden
2.head without beta and gamma
3.use the first half of hidden layer to calculate sim ; use the second half of hidden layer to calculate attention
