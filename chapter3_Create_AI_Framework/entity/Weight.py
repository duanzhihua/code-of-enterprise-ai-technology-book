# -*- coding: utf-8 -*-

class Weight:
    
    
    #第一步：设置和访问Weight的的ID
    def set_weight_index(self, weight_index):
      self._weight_index = weight_index
      
    def get_weight_index(self):
        return self._weight_index
    
    #第二步：设置和访问Weight的来源节点
    def set_from_index(self, from_node_index):
        self._from_index = from_node_index
        
    def get_from_index(self):
        return self._from_index
    

    #第三步：设置和访问Weight的目标节点
    def set_to_index(self, to_node_index):
        self._to_node_index = to_node_index
        
    def get_to_index(self):
        return self._to_node_index
    
    #第四步：设置和访问Weight的值
    def set_value(self, value):
        self._value = value
    
    def get_value(self):
        return self._value
    
