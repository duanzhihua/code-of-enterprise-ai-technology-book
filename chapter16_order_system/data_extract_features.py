# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.readerData import read_data
from data_process.data_rule_adaptation import data_rule_match


def extract_features_order():
    cleaned_df = read_data('./data/rejectionOrders/cleaned_traindata.csv')
    cleaned_df["订单标签"] = cleaned_df.订单号数.apply(lambda x: ('多订单' if (x > 1) else '单订单'))
    '''
    cleaned_df["甩单特征"] = cleaned_df['甩单业务类型'].map(str) + "||" + cleaned_df['甩单备注'].map(str) + "||" + cleaned_df[
        '甩单客户星级'].map(str) + "||" + cleaned_df['甩单政企客户等级'].map(str)
    '''
    cleaned_df["转换前的甩单特征"] = cleaned_df['甩单业务类型'].map(str) + "||" + cleaned_df['甩单备注'].map(str) + "||" + cleaned_df[
        '甩单客户星级'].map(str) + "||" + cleaned_df['甩单政企客户等级'].map(str) + "||" + cleaned_df['甩单工号'].map(str) + "||" + \
                             cleaned_df['甩单工号归属组织id'].map(str) + "||" + cleaned_df['甩单工号归属组织'].map(str) + "||" + \
                             cleaned_df['甩单渠道小类编码'].map(str) + "||" + cleaned_df['甩单渠道大类编码'].map(str) + "||" + \
                             cleaned_df['甩单渠道小类'].map(str) + "||" + cleaned_df['甩单渠道大类'].map(str) + "||" + cleaned_df[
                                 '甩单区局id'].map(str) + "||" + cleaned_df['甩单区局编码'].map(str) + "||" + cleaned_df[
                                 '甩单区局'].map(str) + "||" + cleaned_df['甩单促销工号'].map(str) + "||" + cleaned_df[
                                 '甩单促销部门'].map(str) + "||" + cleaned_df['甩单促销渠道大类编码'].map(str) + "||" + cleaned_df[
                                 '甩单促销渠道小类编码'].map(str) + "||" + cleaned_df['甩单促销渠道大类'].map(str) + "||" + cleaned_df[
                                 '甩单促销渠道小类'].map(str)

    cleaned_df["甩单特征"] = cleaned_df.apply(data_rule_match, axis=1)
    # cleaned_df["甩单特征"] = cleaned_df["转换前的甩单特征"]

    col_n = ['甩单流水号', '订单标签', '甩单特征']
    data_df = pd.DataFrame(cleaned_df, columns=col_n)
    # 划分训练集、验证集、测试集
    num_test = data_df.shape[0] - 1000
    num_train = num_test - 10000
    test_df = data_df.iloc[num_test:]
    dev_df = data_df.iloc[num_train:num_test]
    train_df = data_df.iloc[:num_train]
    test_df.to_csv("./data/rejectionOrders/data.test.csv", encoding='GB18030', index=None, header=None, sep='\t')
    dev_df.to_csv("./data/rejectionOrders/data.val.csv", encoding='GB18030', index=None, header=None, sep='\t')
    train_df.to_csv("./data/rejectionOrders/data.train.csv", encoding='GB18030', index=None, header=None, sep='\t')


if __name__ == '__main__':
    extract_features_order()
