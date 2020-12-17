# -*- coding: utf-8 -*-
import random
import pickle


def data_rule_match(row):
    new_sentence = row['转换前的甩单特征']
    rand_list = [2, 2, 2, 2, 2, 2, 2, 2, 1, 0]

    # 场景一：
    old_phrase: str = "行销全家享||201802-十全十美99元套餐2018版（专营）(2-2A8EM312)|| || ||移动主卡：新装 后付费CDMA ||UIM卡号：********||选号备注：主卡号码随机 ||201801-锦江合作送2500积分促销(2-29O4VMPU) || 201412全家享新融合促销活动，协议期12个月(2-108CS3FI) || || || ||宽带: 已有 ||宽带设备号: ********||备注：如果原宽带设备下的存在IPTV和爱BABY，则全部转入 ||宽带计费速率: 包月制（100M/4M） ||接入方式: FTTH ||201712-全家享新融合99档次宽带加装包（标签网龄）(2-27J39G74) || || ||||备注：新装场景新建分账序号，存量场景使用存量设备分账序号，移动新装号码必须开通翼支付功能和4G功能||"
    new_sentence = mask_from_Bert_paper("[MASK]", "自主研发", rand_list, new_sentence, old_phrase)

    # 场景二：
    old_phrase = "行销全家享||201802-十全十美99元套餐2018版（专营）(2-2A8EM312)|| || ||移动主卡：新装 预付费CDMA ||UIM卡号:********||选号备注:号码随机 ||201801-锦江合作送2500积分促销(2-29O4VMPU)||201412全家享新融合促销活动，协议期12个月(2-108CS3FI)|| || 移动副卡:新装 || 张数:2 张 ||UIM卡号:********||选号备注:副卡号码随机||可选包:201802-不限量5元副卡加装包201802(2-29OGYIBS)||201802-不限量5元副卡功能费201802(2-29IGIOX2)||UIM卡号:********||"
    new_sentence = mask_from_Bert_paper("(test)", "第二种情况", rand_list, new_sentence, old_phrase)

    return new_sentence


def mask_from_Bert_paper(new_phrase_80, new_phrase_10, rand_list, new_sentence, old_phrase):
    rand_index = random.choice(rand_list)
    if old_phrase in new_sentence:
        if rand_index == 2:  # 80% 直接替换为[MASK]
            new_sentence = new_sentence.replace(old_phrase, new_phrase_80)

        if rand_index == 1:  # 10% 转换为一个新单词
            new_sentence = new_sentence.replace(old_phrase, new_phrase_10)

        if rand_index == 0:  # 10% 保留原词  无需转换
            pass
    return new_sentence


def orderNum_id(row_content):
    id_path = './data/orderNum2id.pkl'
    data_id = open(id_path, 'rb')
    business_type_dict = pickle.load(data_id)
    data_id.close()
    return business_type_dict["'" + str(row_content) + "'"]


def type2id(row_content):
    id_path = './data/businessType2id.pkl'
    data_id = open(id_path, 'rb')
    business_type_dict = pickle.load(data_id)
    data_id.close()

    return business_type_dict["'" + str(row_content) + "'"]


def numline_id(row_content):
    id_path = './data/numLines2id.pkl'
    data_id = open(id_path, 'rb')
    num_lines_dict = pickle.load(data_id)
    data_id.close()

    return num_lines_dict["'" + str(row_content) + "'"]


def service_line_id(row_content):
    id_path = './data/serviceAndline2id.pkl'
    data_id = open(id_path, 'rb')
    service_line_dict = pickle.load(data_id)
    data_id.close()

    return service_line_dict["'" + str(row_content) + "'"]


def counter_list(arr):
    result = {}
    for i in set(arr):
        result[i] = arr.count(i)
    return result


def service_line(row):
    service_list = row['业务类型']
    order_num = row['订单号去重数']
    counter_dic = counter_list(service_list)
    contacts = ""
    for key, value in counter_dic.items():
        contact = str(key) + "|" + str(value) + "_"
        contacts = contacts + contact
    contacts = contacts + "#orderNums|" + str(order_num)
    return contacts
