import os
import re
import sys
import random
import time
import requests
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import keras
from keras import backend as K
import tensorflow as tf


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    from dotmap import DotMap
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def get_callback_data(DB_CONFIG):
    """拉取回流标注数据
    """
    sys.path.append(r"../../")
    from utils_toolkit.MySQLHelper import PooledDBConnection
    import csv
    engine = PooledDBConnection(DB_CONFIG)  # 数据库连接对象
    # ignored状态 
    # 0：未标注 1：已忽略 2：待审核 3：审核驳回 4、新建知识 5审核通过 6训练集 7评测集
    sql = f"""
        select question_id, user_question, flow_code
        from oc_annotation_pool 
        where question_id is not null 
        and question_id != ''
        and yn=1
        and dialogue_session_order=1
        and ignored in (5,6,7)
        and create_time like '%2021%'
    """
    res = engine.ExecQuery(sql)

    # with open("datasets/train_callback.csv", "w", encoding="utf-8", newline="") as csvfile:
    #     lines = []
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["knowledge_id", "question", "base_code"])
    #     writer.writerows(res)
    # print("回流标注数据 train_callback.csv 获取完成！")


def create_test_data(test):

    def clean(x):
        """预处理：去除文本的噪声信息"""
        x = re.sub('"', "", x)
        x = re.sub("\s", "", x)  # \s匹配任何空白字符，包括空格、制表符、换页符等
        x = re.sub(",", "，", x)
        return x.strip()
    
    def clean_sim(x):
        """预处理：切分相似问"""
        x = re.sub(r"(\t\n|\n)", "", x)
        x = x.strip().strip("###").replace("######", "###")
        return x.split("###")

    test_data_dict = {}
    for index, row in test.iterrows():
        query = clean(row["user_input"])
        answer_id = clean_sim(str(row["answer_id"]))
        recall = clean_sim(row["recall"])
        recall_id = clean_sim(row["recall_id"])
        # try:
        #     assert len(recall) == len(recall_id) == 10
        # except:
        #     print("len(recall) ≠ len(recall_id) 比如query='拔牙'")
        #     print(index,query,recall,recall_id)
        #     continue
        test_data_dict.setdefault(query, {})
        test_data_dict[query]["kid"] = answer_id
        test_data_dict[query]["candidate"] = recall
        test_data_dict[query]["cid"] = recall_id

    return test_data_dict


def get_nlu_index_from_apollo(config, NAME_SPACE, password_key):
    CONFIG_SERVER_URL = config.apollo.APOLLO_HOST
    APPID = config.apollo.APPID
    CLUSTER_NAME = config.apollo.CLUSTER
    TOKEN = config.apollo.TOKEN
    decrypt_url = config.apollo.DECRYPT_HOST
    api_key = config.apollo.API_KEY
    
    # 从apollo获取NAME_SPACE的配置信息
    url = (
        "{config_server_url}/configfiles/json/{appId}/{clusterName}+{token}/"
        "{namespaceName}".format(
            config_server_url=CONFIG_SERVER_URL,
            appId=APPID,
            clusterName=CLUSTER_NAME,
            token=TOKEN,
            namespaceName=NAME_SPACE,
        )
    )

    res = requests.get(url=url, timeout=10)
    es_index_info = json.loads(res.text)  # dict

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    
    # apollo获取解密后的密码
    headers = {
        "Content-Type": "application/json",
        "X-Gaia-API-Key": api_key,
    }  # X-Gaia-API-Key为PaaS平台上申请的对应key

    with open("/etc/apollo/apollo_private_key", "r") as f:
        PRIVATE_KEY = f.read()

    body = {
        "privateKey": PRIVATE_KEY,
        "cipherText": [es_index_info[password_key]],
    }

    res = requests.post(url=decrypt_url, headers=headers, data=json.dumps(body))
    es_index_info[password_key] = json.loads(res.text)[0]

    return es_index_info


def get_nlu_embedding(text, url):
    """调用 embedding 接口
        获取文本的向量表示
    """
    data = {"text": text}
    headers = {"Content-Type": "application/json", "X-Gaia-Api-Key":"8e21002e-1064-417d-af10-7ac1c4e5601f"}
    res = requests.post(url, json=data, headers=headers, timeout=5)  # 默认超时时间5s
    return res.json().get("data")


class CustomCallback(keras.callbacks.Callback):
    # def __init__(self, encoder, test_data, config, sen2kid, label2vec):
    def __init__(self, encoder, train_data, test_data, config, sen2kid, label2kid):
        self.encoder = encoder
        self.train_data = train_data
        self.test_data = test_data
        self.sen2kid = sen2kid
        self.label2kid = label2kid
        # self.label2vec = label2vec
        self.label2vec = None

    def on_train_begin(self, logs={}):
        self.valid_predictions = []

    def on_epoch_end(self, epoch, logs={}):
        self.label2vec = {
            self.label2kid[label]:vec for label, vec in enumerate(self.encoder.predict(self.train_data)[0])
        }    # { int:[] }

        scores = []
        content = []
        for domain, test_data_dict in self.test_data.items():
            # 模型预测和评估
            n, n1, n2 = 0, 0, 0  # 问题总数、回复的问题数、回复的正确问题数
            nb_kid_not_in_cid = 0
            nb_cid_in_label2vec, nb_cid_not_in_label2vec = 0, 0
            pred_1_list, val_1_list = [], []
            pred_3_list, val_3_list = [], []
            # self.label2vec = self.label2vec_dict[domain]
            for query, info_dict in tqdm(test_data_dict.items()):
                n += 1
                kids = info_dict["kid"]
                candidate = info_dict["candidate"]
                candidate_ids = info_dict["cid"]    # candidate 
                if domain == "test_robust":
                    # 手动添加1条候选（query+噪音）
                    query_noise = query + "？"
                    candidate.append(query_noise)
                    candidate_ids.append(self.sen2kid.get(query_noise, '123'))

                if candidate_ids:
                    n1 += 1
                
                # query & candidate 向量映射
                test_query_vec = get_nlu_embedding(
                    query, "http://10.231.135.106:20182/embedding"
                )  
                all_cand_vecs = []
                for cid in candidate_ids:   # type=str
                    cid = int(cid)
                    if cid not in self.label2vec:
                        nb_cid_not_in_label2vec += 1
                        all_cand_vecs.append(np.zeros(768))
                    else:
                        nb_cid_in_label2vec += 1
                        all_cand_vecs.append(self.label2vec.get(cid, np.zeros(768)))

                # 计算相似度
                # dot_list = np.dot(all_cand_vecs, test_query_vec[0])   # l2之后embedding
                try:
                    dot_list = cosine_similarity(all_cand_vecs, [test_query_vec])   
                except Exception as e:
                    print(e) 
                    continue
                dot_list = [x[0] for x in dot_list]

                # top1预测结果
                max_idx = np.argmax(dot_list)
                pred_one = str(int(candidate_ids[max_idx]))  # 预测的kid
                content.append(
                    "\n".join(
                        list(
                            map(
                                lambda x: str(x),
                                zip(
                                    [query] * len(candidate),
                                    candidate,
                                    candidate_ids,
                                ),
                            )
                        )
                    )
                )
                content.append(
                    "\nmax_idx：{}\tpred_id: {}\tkids: {}".format(
                        max_idx, pred_one, kids
                    )
                )
                if pred_one in kids:
                    n2 += 1
                else:
                    if not set(kids) & set(candidate_ids):
                        nb_kid_not_in_cid += 1

            acc = round(n2 / n, 4)
            p = round(n2 / n1, 4)
            r = round(n1 / n, 4)
            f1 = round(2 * p * r / (p + r), 4)
            scores.append(f1)
            print("\ndomain: {}".format(domain))
            print("问题总数: {}、回复的问题数: {}、回复的正确问题数: {}".format(n, n1, n2))
            print("how much kid not in candidate set: {}".format(nb_kid_not_in_cid))
            print("how much cid not in label2vec: {}".format(nb_cid_not_in_label2vec))
            print("how much cid in label2vec: {}".format(nb_cid_in_label2vec))
            print("acc=n2/n= {}".format(acc))
            print("precision=n2/n1= {}".format(p))
            print("recall=n1/n= {}".format(r))
            print("f1=2pr/(p+r)= {}".format(f1))

