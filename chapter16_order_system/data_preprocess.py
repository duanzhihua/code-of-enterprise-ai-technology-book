# -*- coding: utf-8 -*-
import pickle
from collections import Counter
import numpy as np
import pandas as pd
from entity.node import Node
from data_process.data_rule_adaptation import type2id, numline_id, orderNum_id, service_line, \
    service_line_id
from utils.readerData import read_data


#  data preprocess
def train_data_preprocess():
    raw_order_train_df = read_data('./data/raw_order_train.csv')
    raw_shuaidan_train_df = read_data('./data/raw_rejection_train.csv')

    raw_shuaidan_train_df = raw_shuaidan_train_df.drop_duplicates(subset='甩单流水号', keep=False)
    leftjoin_df = pd.merge(left=raw_shuaidan_train_df, right=raw_order_train_df, on=["甩单流水号"], how="left")
    group_byrejection = leftjoin_df.groupby(leftjoin_df["甩单流水号"])
    rejection_orders = {}
    rejection_orders_list = []
    for rejectionID, group_df in group_byrejection:
        order_type = group_df["业务类型"].values.tolist()
        # 业务类型数据去重
        order_type_unique = list(set(group_df["业务类型"].values.tolist()))
        order_line_id = group_df["订单行项目ID"].values.tolist()
        order_line_id_len = len(group_df["订单行项目ID"].values.tolist())
        order_id = group_df["订单ID"].values.tolist()
        order_id_len = len(group_df["订单ID"].values.tolist())
        order_id_unique = list(set(group_df["订单ID"].values.tolist()))
        order_id_unique_len = len(list(set(group_df["订单ID"].values.tolist())))
        order_id_number = group_df["订单号"].values.tolist()
        orderlen: int = len(group_df["订单号"].values.tolist())
        # 订单号数据去重
        order_id_number_unique = list(set(group_df["订单号"].values.tolist()))
        order_id_number_unique_len = len(list(set(group_df["订单号"].values.tolist())))
        order_node = Node(rejectionID, order_type, order_type_unique, order_line_id, order_line_id_len,
                          order_id, order_id_len, order_id_unique, order_id_unique_len, order_id_number, orderlen,
                          order_id_number_unique, order_id_number_unique_len)
        rejection_orders[rejectionID] = order_node
        rejection_orders_list.append(order_node)
    order_array = np.array(rejection_orders_list)
    order_df = pd.DataFrame(order_array)
    order_df.columns = ['order_node']
    order_df["甩单流水号"] = order_df.order_node.apply(lambda x: x.rejectionID)
    order_df["业务类型"] = order_df.order_node.apply(lambda x: x.orderType)
    order_df["业务类型去重"] = order_df.order_node.apply(lambda x: x.orderType_unique)
    order_df["订单行项目ID"] = order_df.order_node.apply(lambda x: x.orderLineID)
    order_df["订单行项目数"] = order_df.order_node.apply(lambda x: x.orderLineID_len)
    order_df["订单ID"] = order_df.order_node.apply(lambda x: x.orderID)
    order_df["订单ID数"] = order_df.order_node.apply(lambda x: x.orderID_len)
    order_df["订单ID去重"] = order_df.order_node.apply(lambda x: x.orderID_unique)
    order_df["订单ID去重数"] = order_df.order_node.apply(lambda x: x.orderID_unique_len)
    order_df["订单号"] = order_df.order_node.apply(lambda x: x.orderIDNumber)
    order_df["订单号数"] = order_df.order_node.apply(lambda x: x.orderIDNumber_len)
    order_df["订单号去重"] = order_df.order_node.apply(lambda x: x.orderIDNumber_unique)
    order_df["订单号去重数"] = order_df.order_node.apply(lambda x: x.orderIDNumber_unique_len)
    order_lists = order_df["订单号去重数"].values.tolist()
    order_sets = set()
    order_dict = {}
    for num in order_lists:
        order_sets.add(num)
    for i, elems in enumerate(order_sets):
        order_dict["'" + str(elems) + "'"] = i
    save_dict(order_dict, './data/orderNum2id.txt', './data/orderNum2id.pkl')
    order_df["订单号去重数编码"] = order_df.订单号去重数.apply(orderNum_id)
    num_lists = order_df["订单行项目数"].values.tolist()
    num_sets = set()
    num_dict = {}
    for num in num_lists:
        num_sets.add(num)
    for i, elems in enumerate(num_sets):
        num_dict["'" + str(elems) + "'"] = i
    save_dict(num_dict, './data/numLines2id.txt', './data/numLines2id.pkl')
    order_df["订单行项目数编码"] = order_df.订单行项目数.apply(numline_id)
    order_df["业务类型去重"] = order_df.业务类型去重.apply(replace_empty)
    drop_dup_df = order_df.copy()  #
    drop_dup_df.drop_duplicates(subset='业务类型去重', keep='first', inplace=True)
    business_type_lists = drop_dup_df["业务类型去重"].values.tolist()
    business_type_dict = {}
    for i, elems in enumerate(business_type_lists):
        business_type_dict["'" + str(elems) + "'"] = i
    save_dict(business_type_dict, './data/businessType2id.txt', './data/businessType2id.pkl')
    order_df["业务类型去重编码"] = order_df.业务类型去重.apply(type2id)
    order_df["业务类型及行项目数"] = order_df.apply(service_line, axis=1)
    service_line_lists = order_df["业务类型及行项目数"].values.tolist()
    service_line_sets = set()
    service_line_dict = {}
    for num in service_line_lists:
        service_line_sets.add(num)
    for i, elems in enumerate(service_line_sets):
        service_line_dict["'" + str(elems) + "'"] = i
    save_dict(service_line_dict, './data/serviceAndline2id.txt', './data/serviceAndline2id.pkl')
    order_df["业务类型及行项目数编码"] = order_df.业务类型及行项目数.apply(service_line_id)
    # 将甩单表与订单表进行关联，合并成一张表保存。
    traindata_save_df = pd.merge(left=raw_shuaidan_train_df, right=order_df, on=["甩单流水号"], how="left")
    traindata_save_df.to_csv("./data/rejectionOrders/cleaned_traindata.csv", encoding='GB18030')


def compare(s, t):
    return Counter(s) == Counter(t)


def replace_empty(row):
    new_list = []
    for i in row:
        if isinstance(i, str):
            new_list.append(i.replace(" ", ""))
        else:
            new_list.append("NaN")
            print("skip  ", i)

    return " ".join(new_list)


def save_dict(in_dict, filetxt, filepkl):
    typeid_df = pd.DataFrame([in_dict]).T
    typeid_df.to_csv(filetxt, encoding='GB18030', header=None)

    fp = open(filepkl, 'wb')
    pickle.dump(in_dict, fp)
    fp.close()


if __name__ == '__main__':
    train_data_preprocess()
