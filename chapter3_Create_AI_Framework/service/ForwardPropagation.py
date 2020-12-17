# -*- coding: utf-8 -*-
#完成从Input Layer,经过若干层的Hidden Layers,最后得出Output Layer的值
import math
class ForwardPropagation:
    
    def applyForwardPropagation(nodes, weights, instance):
        for i in range(len(nodes)):
            if nodes[i].get_is_bias_unit() == True:
                nodes[i].set_value = 1
                  
        
        # 把数据输入到Input Layer
        #例如说处理instance = [0,1,1]     
        
        for j in range(len(instance) - 1): #训练的时候只需要features
            value_of_feature = instance[j] #获得该条数据中每个Feature具体的值
            
            
            for k in range(len(nodes)):
                
                if j + 1 == nodes[k].get_index(): #索引为0的节点为Bias，所以从索引为1的Node开始
                    nodes[k].set_value(value_of_feature)
        
        
        #Hidden Layer的处理
        for j in range(len(nodes)):
            if nodes[j].get_is_bias_unit() == False and nodes[j].get_level() > 0 :
                
                target_neuron_input = 0 #接受上一个Layer中所有的和自己相关的Neurons和Weights的乘积之和
                target_neuron_output = 0 #经过Non-Linearity后的输出，我们这里使用Sigmoid
                #获得当前Neuron的ID
                target_index = nodes[j].get_index()
                
                for k in range(len(weights)):
                    #获得和当前的Neuron关联的Weight
                    if target_index == weights[k].get_to_index():
                        #获得该Weight的Value
                        weight_value = weights[k].get_value()
                        #获得该Weight的来源的Neuron的ID
                        from_index = weights[k].get_from_index()
                        
                        #获得该来源Neuron的Value
                        for m in range(len(nodes)):
                            #获得该ID的Neuron
                            if from_index == nodes[m].get_index():
                                #获得该Neuron的具体的Value
                               value_from_neuron = nodes[m].get_value()
                               
                               #把Weight和相应的Value相乘，然后累加
                               target_neuron_input = target_neuron_input + (weight_value * value_from_neuron)
                               
                               #一个Weight只连接一个上一个Layer的Neuron，所以不需要继续循环真个神经网络的其它Neurons'
                               break
                             
                #从和Break对其出发，一共按了4次后退键，因为接下来是要应用Sigmoid多当前的Neuron的所有的输入累计后的值进行操作
                target_neuron_output = 1 / (1 + math.exp(- target_neuron_input))
                
                #接下来把输入值和当然Neuron采用NeronSigmoid Activation计算后的值设置进当前的Neuron
                nodes[j].set_input_value(target_neuron_input)
                nodes[j].set_value(target_neuron_output)