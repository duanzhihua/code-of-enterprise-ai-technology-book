
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import csv
import os
import codecs
import json
import random
import logging
import argparse
from tqdm import tqdm, trange
import pandas as pd
from sklearn import metrics
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from run_classifier_order_system import MyPro
from run_classifier_order_system import convert_examples_to_features
from utils.readerData import read_data
from data_process.data_rule_adaptation import data_rule_match

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding('GB18030') #utf-8 encoding='GB18030'
    is_py3 = False

file_name="predict_test.csv"
prdict_file_name= "./data/rejectionOrders/"+file_name

def main():
    # ArgumentParser对象保存了所有必要的信息，用以将命令行参数解析为相应的python数据类型
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_lower_case",
                        default = False,
                        action = 'store_true',
                        help = "英文字符的大小写转换，对于中文来说没啥用")
    # other parameters
    parser.add_argument("--max_seq_length",
                        default = 22,
                        type = int,
                        help = "字符串最大长度")
    parser.add_argument("--eval_batch_size",
                        default = 1,
                        type = int,
                        help = "验证时batch大小")
    # 调用add_argument()向ArgumentParser对象添加命令行参数信息，这些信息告诉ArgumentParser对象如何处理命令行参数
    parser.add_argument("--data_dir",
                        default = 'data/rejectionOrders',
                        type = str,
                        #required = True,
                        help = "The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model",
                        default = 'bert-base-chinese',
                        type = str,
                        #required = True,
                        help = "choose [bert-base-chinese] mode.")
    parser.add_argument("--task_name",
                        default = 'MyPro',
                        type = str,
                        #required = True,
                        help = "The name of the task to train.")
    parser.add_argument("--output_dir",
                        default = 'checkpoints/',
                        type = str,
                        #required = True,
                        help = "The output directory where the model checkpoints will be written")
    parser.add_argument("--model_save_pth",
                        default = 'checkpoints/bert_classification.pth',
                        type = str,
                        #required = True,
                        help = "The output directory where the model checkpoints will be written")

    parser.add_argument("--no_cuda",
                        default = False,
                        action = 'store_true',
                        help = "用不用CUDA")
    parser.add_argument("--local_rank",
                        default = -1,
                        type = int,
                        help = "local_rank for distributed training on gpus.")
    parser.add_argument("--seed",
                        default = 777,
                        type = int,
                        help = "初始化时的随机数种子")
    parser.add_argument("--gradient_accumulation_steps",
                        default = 1,
                        type = int,
                        help = "Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimize_on_cpu",
                        default = False,
                        action = 'store_true',
                        help = "Whether to perform optimization and keep the optimizer averages on CPU.")
    parser.add_argument("--fp16",
                        default = False,
                        action = 'store_true',
                        help = "Whether to use 16-bit float precision instead of 32-bit.")
    parser.add_argument("--loss_scale",
                        default = 128,
                        type = float,
                        help = "Loss scaling, positive power of 2 values can improve fp16 convergence.")

    args = parser.parse_args()

    # 对模型输入进行处理的processor
    processors = {'mypro': MyPro}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
       cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),num_labels=len(label_list))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(args.model_save_pth)['state_dict'])
     #加入数据预处理。
    raw_test_df = read_data('./data/raw_test.csv')
    raw_test_df["转换前的甩单特征"] = raw_test_df['甩单业务类型'].map(str) + "||" + raw_test_df['甩单备注'].map(str) + "||" + raw_test_df[
        '甩单客户星级'].map(str) + "||" + raw_test_df['甩单政企客户等级'].map(str) + "||" + raw_test_df['甩单工号'].map(str) + "||" + \
                              raw_test_df['甩单工号归属组织id'].map(str) + "||" + raw_test_df['甩单工号归属组织'].map(str) + "||" + \
                              raw_test_df['甩单渠道小类编码'].map(str) + "||" + raw_test_df['甩单渠道大类编码'].map(str) + "||" + \
                              raw_test_df['甩单渠道小类'].map(str) + "||" + raw_test_df['甩单渠道大类'].map(str) + "||" + \
                              raw_test_df['甩单区局id'].map(str) + "||" + raw_test_df['甩单区局编码'].map(str) + "||" + \
                              raw_test_df['甩单区局'].map(str) + "||" + raw_test_df['甩单促销工号'].map(str) + "||" + raw_test_df[
                                  '甩单促销部门'].map(str) + "||" + raw_test_df['甩单促销渠道大类编码'].map(str) + "||" + raw_test_df[
                                  '甩单促销渠道小类编码'].map(str) + "||" + raw_test_df['甩单促销渠道大类'].map(str) + "||" + raw_test_df[
                                  '甩单促销渠道小类'].map(str)

    raw_test_df["甩单特征"] = raw_test_df.apply(data_rule_match, axis=1)
    # raw_test_df["甩单特征"] = raw_test_df["转换前的甩单特征"]

    raw_test_df['index'] = raw_test_df.index
    col_n = ['甩单流水号',  '甩单特征']
    raw_data_df = pd.DataFrame(raw_test_df, columns=col_n)
    predict(model, processor, args, label_list, tokenizer, device, raw_data_df)


def predict(model, processor, args, label_list, tokenizer, device, raw_data_df):
    test_examples = processor.get_test_examples(args.data_dir, file_name=file_name)
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids )
    # Run prediction for full data
    test_dataloader = DataLoader(test_data,  batch_size=args.eval_batch_size)

    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    for input_ids, input_mask, segment_ids  in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            pred = logits.max(1)[1]
            predict = np.hstack((predict, pred.cpu().numpy()))
        logits = logits.detach().cpu().numpy()
    predict_df = pd.DataFrame(predict)
    predict_df['index'] = predict_df.index
    predict_df.columns = ['标签预测','index']

    raw_data_df['index'] = raw_data_df.index
    raw_data_df.columns = ['甩单流水号', '甩单特征', 'index']

    Predict_result_df = pd.merge(left=raw_data_df, right=predict_df, on=["index"], how="left")
    Predict_result_df =Predict_result_df[Predict_result_df["标签预测"] ==1 ]
    Predict_result_df = pd.DataFrame(Predict_result_df,  columns=["甩单流水号" ])
    Predict_result_df.to_csv("./data/output/predict_result.csv", encoding='GB18030', index=None,   header=None)


if __name__ == '__main__':
  main()






















