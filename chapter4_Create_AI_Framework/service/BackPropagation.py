# -*- coding: utf-8 -*-
from service.ForwardPropagation import ForwardPropagation

#完成Deep Learning Framework中最为核心的功能：Back Propagation:
#第一步： 从误差的结果出发，从最右侧到最左侧（不包含Input Layer）遍历整个Neuron Network构成的Chain；
#第二步：在遍历的时候计算每个Neuron对误差结果应该负的责任；
#第三步：进行Derivative计算；
#第四步：通过Gradient Desendent来调整Weights的值以减少预测的误差
class BackPropagation:
    
    def applyBackPragation(instances, nodes, weights, learning_rate):
        
        num_of_features = len(instances[0]) - 1 #记录输入的Features的个数，instance的最后一列是Real Result
                
        #循环遍历所有的Training Dataset 完成一个Epoch 并进行每个节点所负责的Error的记录
        for i in range(len(instances)):
            
            #使用Forward Propagation从Input Layer出发，经过Hidden Layers,最后获得Output
            nodes = ForwardPropagation.applyForwardPropagation(nodes, weights, instances[i])
            predicted_value = nodes[len(nodes) - 1].get_value() #记录该次Forward Propagation最终的误差
            
            
            actual_value = instances[i][num_of_features] #获得当前instance的Real Value
            
            minor_error = predicted_value - actual_value #计算预测值和真实值之间的误差
            
            nodes[len(nodes)-1].set_minor_error(minor_error) #把该误差值设置进Output Layer中的输出节点中
            
            #因为输出节点已经计算完误差，所以会减掉2；
            #因为Input Layer不参与计算，所以range的三个参数中的第二个参数是num_of_features
            #该循环遍历是从Output Layer的前面的一个Hidden Layer开始的，或者说是从最后一个Hidden Layer开始的
            for j in range(len(nodes)-2, num_of_features, -1):
                target_index = nodes[j].get_index() #从最后一个Hidden Layer的最后一个Neuron开始计算，然后依次向前
                
                sum_minor_error = 0 #存储当前Neuron应该为误差所要负的责任
                
                #循环遍历所有的Weights以获得以target_index为出发点的所有Weights
                for k in range(len(weights)): 
                    #如果当前的Weight是以target_index所在的Neuron为出发节点，则说明该Weight需要多结果负(直接)责任 
                    if weights[k].get_from_index() == target_index:
                         
                         affecting_theta = weights[k].get_value() #获得当前Weight的Value
                         
                         affected_minor_error = 1 #初始化当前Neuron对结果影响的Value
                         
                         target_minor_error_index = weights[k].get_to_index() #计算当前Neuron所影响的下一个Layer中具体的Neuron的ID
                         
                         for m in range(len(nodes)):
                             if nodes[m].get_index() == target_minor_error_index:
                                 affected_minor_error = nodes[m].get_minor_error()
                            
                         #获得当前Weight的触发Neuron对结果负责任的具体的值
                         updated_minor_error = affecting_theta * affected_minor_error
                         
                         #把对下一个Layer中具体误差负责任的所有误差都累加到当前Neuron并保存到当前的Neuron中
                         sum_minor_error = sum_minor_error + updated_minor_error
                #保存当前的Neuron对下一个Layer的所有的Neurons所造成的Loss影响的总和                
                nodes[j].set_minor_error(sum_minor_error)
        
        #这里是对我们在ForwardPropagation使用的是Sigmoid Activation，所以这里是对Sigmoid进行求导
        # 然后更新Weights！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            for j in range(len(weights)):
                weight_from_node_value = 0
                weight_to_node_value = 0
                weight_to_node_error = 0
                
                for k in range(len(nodes)):
                  
                    if nodes[k].get_index() == weights[j].get_from_index():
                        weight_from_node_value = nodes[k].get_value()
                    
                    if nodes[k].get_index() == weights[k].get_to_index():
                    
                        #weight_to_node_value == nodes[k].get_value()
                        weight_to_node_value = nodes[k].get_value()
                        weight_to_node_error = nodes[k].get_minor_error()
                        
                
                #进行求导，因为我们在ForwardPropagation使用的是Sigmoid Activation，所以这里是对Sigmoid进行求导
                # Forward Propagation中的Sigmoid代码：target_neuron_output = 1 / (1 + math.exp(- target_neuron_input))
                
                derivative = weight_to_node_error * (weight_to_node_value * (1 - weight_to_node_value
                                                                             )) * weight_from_node_value
                
                #更新Weight，这是下次计算的时候能够更加准确的关键，因为把上面的计算成果运用在了调整Neuron Network的Weights上                                                            
                weights[j].set_value(weights[j].get_value() - derivative * learning_rate)         
           
        
        return nodes, weights
        
