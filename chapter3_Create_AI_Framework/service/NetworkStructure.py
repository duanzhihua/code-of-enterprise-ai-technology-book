# -*- coding: utf-8 -*-

#导入要使用的Node类
from entity.Node import Node 
class NetworkStructure:
    
    #创建整个神经网络的所有节点
    def create_nodes(num_of_features, hidden_layers):
        nodes = []
        
        nodeIndex = 0 #Neuron的ID
        
        #Input layer
        
        #Bias Unit
        
        node = Node()
        node.set_level(0)
        node.set_index(nodeIndex)
        node.set_label("+1")
        node.set_is_bias_unit(True)
        nodes.append(node)
        nodeIndex = nodeIndex + 1
        
        print(node.get_label(), "\t", end = '')
        
        
        for i in range(num_of_features):
            print("V" + str(i+1) + "\t", end = '')
            node = Node()
            node.set_level(0)
            node.set_index(nodeIndex)
            node.set_label("+1")
            node.set_is_bias_unit(False)
            nodes.append(node)
            nodeIndex = nodeIndex + 1
        
        print("")
        
        #Hidden layer
        for i in range(len(hidden_layers)):
            print("Hidden layer creation: ", end = '')
            
            #Bias Unit
        
            node = Node()
            node.set_level(i+1)
            node.set_index(nodeIndex)
            node.set_label(i+1)
            node.set_is_bias_unit(True)
            nodes.append(node)
            nodeIndex = nodeIndex + 1
            
            print(node.get_label(), "\t", end = '')
            
            #创建该layer的Neurons
            for j in range(hidden_layers[i]):
                #创建该layer内部的Neuron
                node = Node()
                node.set_level(i+1)
                node.set_index(nodeIndex)
                node.set_label("N[" + str(i+1) + "][" + str(j + 1) + "]")
                node.set_is_bias_unit(False)
                nodes.append(node)
                nodeIndex = nodeIndex + 1
                
                print(node.get_label(), "\t", end = '')
                
                
            print("")
        
        print("")
        
        #Output layer
        node = Node()
        node.set_level(1 + len(hidden_layers))
        node.set_index(nodeIndex)
        node.set_label("Output")
        node.set_is_bias_unit(False)
        nodes.append(node)
        nodeIndex = nodeIndex + 1
        print("Output layer: ", node.get_label())
        
        return nodes


