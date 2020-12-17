# -*- coding: utf-8 -*-

class Node:
    
    #第一步：设置和访问Neuron在全局所处的Layer层
    
    #设置Neuron在整个神经网络中唯一的Layer层
    def set_level(self, level):
        self._level = level
        
    #获得Neuron在整个神经网络中唯一的Layer层
    def get_level(self):
        return self._level
    
    #第二步：设置和访问Neuron全局唯一的ID
    
    #设置Neuron在整个神经网络中唯一的ID
    def set_index(self, index):
        self._index = index
    
    #获得Neuron在整个神经网络中唯一的ID
    def get_index(self):
        return self._index
    
    #第三步：设置和访问Neuron的Label名称

    
    #设置Neuron在整个神经网络Label名称
    def set_label(self, label):
        self._label = label
    
    #获得Neuron在整个神经网络Label名称
    def get_label(self):
        return self._label
    
    
    #第四步：判断当前的Neuron是否是一个Bias

    
    #设置Neuron在整个神经网络中是否是一个Bias
    def set_is_bias_unit(self, is_bias_unit):
        self._is_bias_unit = is_bias_unit
    
    #获得Neuron在整个神经网络中是否是一个Bias
    def get_is_bias_unit(self):
        return self._is_bias_unit
    
    
    
    
    
    
    

