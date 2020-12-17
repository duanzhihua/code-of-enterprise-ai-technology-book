# -*- coding: utf-8 -*-
from entity.Weight import Weight
import math
import random

class NetworkConnection:
    
    #Weight代表了神经网络中前后层次之间的联系
    def create_Weights(nodes, num_of_features, hidden_layers):
        weights = []
        
        #整个Neural Network的层数= Input layers + Hidden layers + Output layers
        total_layers = 1 + len(hidden_layers) + 1 
        
        #记录Weight的ID
        weight_index = 0
        
        #Weight发生于前后两个Layers之间，所以要从总的Layer数量中减去1
        for i in range(total_layers - 1): 
            #循环遍历所有的节点
            
            for j in range(len(nodes)):
                #判断当前节点所在的具体的Layer
                

            
                if nodes[j].get_level() == i:
                    #再次循环遍历所有的节点，核心目的在判断不同节点之间的Layer的先后关系
                    for k in range(len(nodes)):
                        
                        #构成Weight的前提条件是有前后相邻的Layer关系
                        if nodes[k].get_level() == i + 1:
                            #比较的Node之间不能是Bias成员
                            #当我们的设置hidden_layers = [4,2]的时候ID为0、3、8的Node为Bias
                            if nodes[k].get_is_bias_unit() == False:
                                if nodes[j].get_is_bias_unit() == False:
                                    #节点所在的Layer的ID越小，就越在前面，nodes[j]在第i层，而nodes[k]在第i+1层
                                    #从nodes[j]触发到nodes[k]之间创建Weight
                                    weight = Weight()
                                    weight.set_weight_index(weight_index)
                                    weight.set_from_index(nodes[j].get_index())
                                    weight.set_to_index(nodes[k].get_index())
                                    
                                    #下面是5节课内从零起步(无需数学和Python基础)编码实现AI框架
                                    #第一节课：从零起步编码实现多层次神经网络 的最大难点
                                    # 创建Weight的之Value，此时请再次运行http://playground.tensorflow.org/
                                    # 会发现AI运行所有的故事都是在更新Weight的值                                    
                                    
                                    #具体算法实现请参考本课程群内部的文件
                                    range_min = 0
                                    range_max = 1
                                    
                                    init_epsion = math.sqrt(6) / (math.sqrt(num_of_features) + 1)
                                    
                                    rand = range_min + (range_max - range_min) * random.random()
                                    rand = rand * (2 * init_epsion) -init_epsion
                                    
                                    weight.set_value(rand)
                                    
                                    
                                    weights.append(weight) #加入到weights集合中
                                    
                                    weight_index = weight_index + 1
                                    #print("The weight from " + str(nodes[j].get_index()) + " to "+ str(nodes[k].get_index()) + " : " + str(rand))
                                    print("The weight from " + str(nodes[j].get_index()) + " at layers[" + str(nodes[j].get_level()) + "] to "+ 
                                          str(nodes[k].get_index()) + " at layers[" + str(nodes[k].get_level()) + "] : " + str(rand))
                            
        
        
        
        return weights
    
