# -*- coding: utf-8 -*-
from service.ForwardPropagation import ForwardPropagation

#Loss的计算方法一：最简单的Loss的计算方式是把每个Instance计算的Predicted的结果减去Real Value然后累计，完毕后除以所有的Instances的个数
#方法一的优点：方便理解；
#方法一的缺点：不具有实用价值，没有真实的AI框架采用这种算法；
#Loss的计算的方法二： 把每个Instance计算的Predicted的结果减去Real Value后进行绝对值求值，然后累计，完毕后除以所有的Instances的个数
#方法二的优点：真实反馈出预测值和真实值的差异
#方法二的缺点： 可能获得很慢的Converge的过程
#Loss的计算的方法三： 把每个Instance计算的Predicted的结果减去Real Value后进行平方的求值，然后累计，完毕后除以所有的Instances的个数
#方法三的缺点： 当数据量很大的时候，会导致很大的计算量；并且有些计算是不必要的；
#方法四：采用样本方差的计算，该方法适合数据量比较大的时候

class LossComputation:
    
    def compute_loss(instances, nodes, weights):
        
        
        total_loss = 0 #记录累加的误差
        
        for i in range(len(instances)):
            
            instance = instances[i]
            
            nodes = ForwardPropagation.applyForwardPropagation(nodes, weights, instance)
            
            predicted_value = nodes[len(nodes) - 1].get_value();
            real_value = instance[len(instance) - 1]
            
            squared_error = (predicted_value - real_value) * (predicted_value - real_value)
            
            total_loss = total_loss + squared_error
        
        total_losss = total_loss / len(instances)
        
        return total_losss
        