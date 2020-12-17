# -*- coding: utf-8 -*-


class Node:
    def __init__(self, rejection_id, order_type, order_type_unique, order_line_id, order_line_id_len,
                 order_id, order_id_len, order_id_unique, order_id_unique_len, order_id_number, order_id_number_len,
                 order_id_number_unique, order_unique_len):
        self.rejectionID = rejection_id
        self.orderType = order_type
        self.orderType_unique = order_type_unique
        self.orderLineID = order_line_id
        self.orderLineID_len = order_line_id_len
        self.orderID = order_id
        self.orderID_len = order_id_len
        self.orderID_unique = order_id_unique
        self.orderID_unique_len = order_id_unique_len
        self.orderIDNumber = order_id_number
        self.orderIDNumber_len = order_id_number_len
        self.orderIDNumber_unique = order_id_number_unique
        self.orderIDNumber_unique_len = order_unique_len
