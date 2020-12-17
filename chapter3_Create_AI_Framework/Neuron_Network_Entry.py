# -*- coding: utf-8 -*-
from service.NetworkStructure import NetworkStructure
from service.NetworkConnection import NetworkConnection
from service.ForwardPropagation import ForwardPropagation
#from entity.Node import Node
#Exclusive OR: 只有第一个元素和第二个元素不同的时候，结果才是1，否则为0
instances = [[0,0,0],
             [0,1,1],
             [1,0,1],
             [1,1,0]]

num_of_features = len(instances[0]) - 1 #在这里是只有第一列和第二列是features，第三列是根据某种关系而得出的结果

#hidden_layers = [4,2] #这里hardcode为两个Hidden layer，第一个Hidden Layer有4个Neuron，第二个Hidden Layer有连个Neuron
hidden_layers = [8,4,2]
nodes = NetworkStructure.create_nodes(num_of_features,hidden_layers)
   
#ID为0、3、8的Node为Bias
#for i in range(len(nodes)):
#    print("This is a bias:" + str(nodes[i].get_is_bias_unit()))

weights = NetworkConnection.create_Weights(nodes,num_of_features, hidden_layers)

#从Input Layer触发，经过所有的Hidden Layers的处理，最终得出Output Layer的结果。

for i in range(len(instances)):
    instance = instances[i]
    
    ForwardPropagation.applyForwardPropagation(nodes, weights, instance)
    
    #得出此次Forward Propagation的输出结果
    #print("Prediction: " + str(nodes[len(nodes) - 1].get_value()))

    print("Prediction: " + str(nodes[len(nodes) - 1].get_value()) + 
          " while real value is: " + str(instance[num_of_features]))    
