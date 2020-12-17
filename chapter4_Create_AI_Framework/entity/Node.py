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
    
    #获得上一个Layer中所有和自己有Weight关系的Neron的Values并设置给当然的Neuron
    def set_input_value(self, input_value):
        self._input_value = input_value
    
    #获得当前Neuron中接收到的上一个Layer中所有和自己有Weight关系的Neron的Values
    def get_input_value(self):
        return self._input_value
    
    #当前的Neuron多接受到的输入源的所有数据进行Non-Linearity的处理后会产生具体的结果
    #接下来使用setter和getter把来存储和访问该非线性计算结果
    
    #获得经过非线性计算后的结果并设置给当前的Neuron
    def set_value(self, value):
        self._value = value
    
    #获得当前Neuron的Value
    def get_value(self):
        return self._value
    
    #设置当前Neuron对刚刚发生的Forward Propagation而导致的Loss的具体责任
    def set_minor_error(self, minor_error):
        self._minor_error = minor_error
        
    #获取当前Neuron对刚刚发生的Forward而导致的Loss的具体责任
    def get_minor_error(self):
        return self._minor_error
    
    
    
    

