# -*- coding: utf-8 -*-
class FeatureNormalization:
    
    
    def normalize(instances):
        num_of_elements = len(instances[0])
        
        max_items = []
        min_items = []
        
        #求出每一个Feature（在这里也包含了对结果的正则化操作，这样才能保证误差的真实性）的最大值和最小值
        for j in range(num_of_elements):
            
            temp_max = instances[0][j]
            temp_min = instances[0][j]
            
            #进行正则化的时候是针对Input Source的每一列数据进行的操作
            for i in range(len(instances)):
                instance = instances[i][j]
                
                if instance > temp_max:
                    temp_max = instance
                    
                if instance < temp_min:
                    temp_min = instance
                    
            max_items.append(temp_max)
            min_items.append(temp_min)
            
            #初始化状态必须是数据源里面的数据
            #temp_max = instances[0][j]
            #temp_min = instances[0][j]
            
        #The process of normalization
        for i in range(len(instances)):
            for j in range(num_of_elements):
                value = instances[i][j]
                
                #获取当前行的元素所在的列的最大值和最小值
                maxItem = max_items[j]
                minItem = min_items[j]
                
                #正则化到符合Sigmoid函数的定义域[-4,-4]和值域[0,1]
                if j == num_of_elements - 1:
                    newMax = 1
                    newMin = 0
                    
                else:
                    newMax = 4
                    newMin = -4
                    
                value = ((newMax - newMin)*((value - minItem) /(maxItem - minItem))) + newMin
                
                instances[i][j] = value
                
        return instances
            
