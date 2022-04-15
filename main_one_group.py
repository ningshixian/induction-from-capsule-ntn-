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

import keras
from keras import backend as K
from keras.layers import *
from keras.layers.core import Lambda
from keras.models import Model
from keras.optimizers import *
from keras.engine.topology import Layer
from keras import activations
from keras.utils.np_utils import to_categorical

import tensorflow as tf

from utils import (
    get_config_from_json,
    get_callback_data,
    create_test_data,
    get_nlu_index_from_apollo,
    get_nlu_embedding,
    CustomCallback,
)

sys.path.append(r"../")
from Capsule_Keras import Capsule

"""
nohup python main_one_group.py > log.txt 2>&1 &
"""


# sets random seed
seed = 123
random.seed(seed)
np.random.seed(seed)

# # 使用 CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 使用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# set GPU memory
# 方法1:显存占用会随着epoch的增长而增长,也就是后面的epoch会去申请新的显存,前面已完成的并不会释放,为了防止碎片化
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需求增长
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
K.set_session(sess)

# specify the batch size and number of epochs
LR = 2e-5  # 用足够小的学习率[3e-4, 5e-5, 2e-5] (adam默认学习率是0.001)
DROPOUT_RATE = 0.5  # 0.3 0.5
BATCH_SIZE = 1  # 1 64
EPOCHS = 10
optimizer = "adam"
opt = Adam(lr=LR, clipvalue=1.0)
kid2label, label2kid = {}, {}  # kid转换成递增的id格式
# Capsule nn
routings = 3
HIDDEN_DIM = 64     # 降维


# def neural_tensor_layer(x, out_size=100):
#     """neural tensor layer (NTN)
#     v = f(cMe)
#     r=sigmoid(Wv+b)
#     """
#     class_vector, query_encoder = x
#     B, C, H = class_vector.shape
#     # print("class_vector shape:", class_vector.shape)
#     # print("query_encoder shape:", query_encoder.shape)
#     M = tf.compat.v1.get_variable("M", shape=[H, H, out_size], dtype=tf.float32)  #, initializer=keras.initializers.glorot_normal())
#     mid_pro = []
#     for slice in range(out_size):
#         slice_inter = tf.matmul(tf.matmul(class_vector, M[:, :, slice]), query_encoder, transpose_b=True)  # (C,H)*(H,H)→(C,H) *(Q,H).T→(C,Q)
#         mid_pro.append(slice_inter)
#     tensor_bi_product = tf.concat(mid_pro, axis=0)  # (C*K,Q)
#     V = tf.nn.relu(tf.transpose(tensor_bi_product))  # (Q,C*K)

#     W = tf.compat.v1.get_variable("w", [C * out_size, C], dtype=tf.float32)
#     b = tf.compat.v1.get_variable("b", [C], dtype=tf.float32)
#     probs = tf.nn.sigmoid(tf.matmul(V, W) + b)  # (Q,C*K)*(C*K,C)→(Q,C)
#     return probs


class ntn_layer(Layer):
    def __init__(self, out_size=10, activation="relu", **kwargs):   # out_size=100
        super(ntn_layer, self).__init__(**kwargs)
        self.out_size = out_size  # 张量参数的个数（k）
        self.activation = activations.get(activation)
        self.test_out = 0

    def build(self, input_shape):
        super(ntn_layer, self).build(input_shape)
        # print(input_shape)  # [(None, C, 768), (None, yy, 768)]
        B, C, H = input_shape[0]
        self.W = self.add_weight(
            name="w",
            shape=[H, H, self.out_size],
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b", shape=(C,), initializer="zeros", trainable=True
        )
        self.U = self.add_weight(
            name="u",
            shape=(C * self.out_size, C),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, x, mask=None):
        class_vector, query_encoder = x
        batch_size = K.shape(class_vector)[0]
        self.C = int(class_vector.shape[1])
        self.Q = int(query_encoder.shape[1])

        V_out, h, mid_pro = [], [], []
        for i in range(self.out_size):  # computing the innner products
            # temp = K.dot(class_vector, self.W[:,:,i])
            # h = K.sum(temp*query_encoder, axis=1)
            # mid_pro.append(h)   # +self.b[i]
            slice_inter = tf.matmul(
                tf.matmul(class_vector, self.W[:, :, i]),
                query_encoder,
                transpose_b=True,
            )  # (C,H)*(H,H)→(C,H) *(Q,H).T→(C,Q)
            mid_pro.append(slice_inter)

        tensor_bi_product = K.concatenate(mid_pro, axis=1)  # axis=0
        # print(tensor_bi_product.shape)   # (C*切片数,Q)
        print(
            "{}*{}→(C,H)*{}→{}".format(
                class_vector.shape,
                self.W[:, :, 0].shape,
                K.transpose(query_encoder).shape,
                K.int_shape(slice_inter),
            )
        )

        V = self.activation(tf.transpose(tensor_bi_product, [0, 2, 1]))
        # V = self.activation(K.reshape(tensor_bi_product, (self.out_size, batch_size)))
        tensor_bi_product = tf.matmul(V, self.U) + self.b
        tensor_bi_product = keras.activations.sigmoid(
            tensor_bi_product
        )  # (Q,C*切片数)*(C*切片数,C)→(Q,C)
        print("{}*{}→{}".format(V.shape, self.U.shape, tensor_bi_product.shape))

        self.test_out = K.shape(tensor_bi_product)
        return tensor_bi_product

    def compute_output_shape(self, input_shape):
        return (None, self.Q, self.C)


def consine_distance(vectors):
    (class_vector, query_vector) = vectors  # [(None, C, 768), (None, yy, 768)]
    class_vector = K.l2_normalize(class_vector, axis=1)
    query_vector = K.l2_normalize(query_vector, axis=1)

    class_vector = tf.transpose(class_vector, [0,2,1])
    # print(tf.matmul(query_vector, class_vector).shape)
    similar_scores = tf.matmul(query_vector, class_vector)  # (Q,H)*(C,H).T→(Q,C)
    return similar_scores


class DataLoader:
    def __init__(self, config):
        super().__init__()
        self.kid2label = {}
        self.config = config
        DB_CONFIG = get_nlu_index_from_apollo(config, "slot", "passwd")
        # get_callback_data(DB_CONFIG)    #拉取最新的回流数据用于训练；

        # load Longfor ROBOT training dataset.
        # # df.columns = ['knowledge_id','question','base_code','category_id']
        df = pd.read_csv(
            self.config.data_loader.train_file,
            header=0,
            sep=",",
            encoding="utf-8",
            engine="python",
            # nrows=256,      # 16384 256
        )
        # 过滤数据，只保留 flow_code=LXHRSDH
        rs_base_list = ["RSNBBASE", "RSZXBASE", "RSWBBASE", "RENSHIBASE", "RSBASENEW"]
        df = df[(df.base_code.isin(rs_base_list))]

        self.sentences, self.y_train = (
            np.array(df["question"]),
            np.array(df["knowledge_id"]),
        )
        self.sen2kid = {
            row["question"]: row["knowledge_id"] for idx, row in df.iterrows()
        }

        self.X_train = self.sentences

    def get_train_data(self):
        train_inputs = []
        train_inputs.append(self.X_train)
        train_inputs.append(self.y_train)
        return train_inputs, self.sen2kid

    def get_test_data(self):
        test_data_list = {}

        domain = "test_acc"
        df_test = pd.read_csv(
            self.config.data_loader.test_file1,
            header=0,
            sep=",",
            encoding="utf-8",
            engine="python",
        )  # 防止乱码
        # print(df_test.shape)
        test_data_list[domain] = create_test_data(df_test)
        print("domain: {}, len: {}".format(domain, len(test_data_list[domain])))  # 1906

        domain = "test_robust"
        df_test = pd.read_csv(
            self.config.data_loader.test_file2,
            header=0,
            sep=",",
            encoding="utf-8",
            engine="python",
        )  # 防止乱码
        test_data_list[domain] = create_test_data(df_test)
        print("domain: {}, len: {}".format(domain, len(test_data_list[domain])))  # 966

        return test_data_list


# Define a slice layer using Lamda layer
def slice(x, h1, h2, w1, w2):
    """Define a tensor slice function"""
    return x[:, h1:h2, w1:w2]


class C_Model(object):
    def __init__(self, config):
        super(C_Model, self).__init__()
        # self.train_model = self.build_model()     # 返回方法本身
        self.class_vector = None

    def build_model(self, C1, K1, Q1, num_sen, hidden_size=768):
        """_summary_

        Args:
            C1 (_type_): num_classes
            K1 (_type_): 支持集的数量=num_sen*0.8
            Q1 (_type_): 查询集的数量=num_sen*0.2
            hidden_size (int, optional): vector dimension. Defaults to 768.
        """
        input_vec = Input(shape=(None, hidden_size))  # None / num_sen
        # input_vec = Input(shape=(C1*(K1+Q1), hidden_size))  # 凑整 (k*c,seq_len,emb_size)
        print("input_vec.shape: ", input_vec.shape)  # (?, num_sen, 768)

        # # 划分支持集和查询集
        # # tf.slice 只能处理 rank=2 的矩阵操作！！
        # support_encoder = Lambda(lambda x: tf.slice(x, [0, 0],
        #                             [C1 * K1, hidden_size]), name="support_set")(input_vec)
        # query_encoder = Lambda(lambda x: tf.slice(x, [C1 * K1, 0],
        #                             [C1 * Q1, hidden_size]), name="query_set")(input_vec)
        support_encoder = Lambda(
            slice,
            arguments={"h1": 0, "h2": K1, "w1": 0, "w2": hidden_size},
            name="support_set",
        )(input_vec)
        query_encoder = Lambda(
            slice,
            arguments={"h1": K1, "h2": num_sen, "w1": 0, "w2": hidden_size},
            name="query_set",
        )(input_vec)
        print("support_encoder shape:", support_encoder.shape)  # (?, xx, 768)
        print("query_encoder shape:", query_encoder.shape)  # (?, yy, 768)
        # support_encoder = Dense(HIDDEN_DIM)(support_encoder)    # 降维

        # 类表示学习
        # Capsule 层输出张量形状 (batch, num_capsule, dim_capsule)
        capsule_1 = Capsule(num_capsule=C1, dim_capsule=HIDDEN_DIM, routings=3, name="capsule_layer")
        dense_1 = Dense(hidden_size, activation=None, name="dense_layer")

        self.class_vector = capsule_1(support_encoder)  # v-[bs,类别数,类表示维度]
        print("class_vector.shape:", self.class_vector.shape)  # (?, 23, 64)
        self.class_vector = dense_1(self.class_vector)
        print("class_vector.shape:", self.class_vector.shape)  # (?, 23, 768)

        # 关系建模层
        probs = Lambda(consine_distance)([self.class_vector, query_encoder])  # (?,yy,C)
        # probs = ntn_layer()([self.class_vector, query_encoder])  # (?,yy,C)

        # 训练模型构造
        model = Model(inputs=input_vec, outputs=probs)
        model.compile(
            loss=lambda y_true, y_pred: y_true * K.relu(0.9 - y_pred) ** 2
            + 0.25 * (1 - y_true) * K.relu(y_pred - 0.1) ** 2,  # mean_squared_error loss
            optimizer=opt,
            metrics=["accuracy"],
        )
        model.summary()

        # 编码器构造, 用于推理阶段
        output = capsule_1(input_vec)
        output = dense_1(output)
        encoder = Model(inputs=input_vec, outputs=output)

        return model, encoder


class Trainer:
    def __init__(self, model, data_loader, config):
        data, sen2kid = (
            data_loader.get_train_data()
        )  # data=[question, knowledge_id] {question:knowledge_id}
        self.data = data
        self.config = config
        self.model = model

        self.sen2kid = sen2kid
        self.test_data = data_loader.get_test_data()
        self.es_index_info = get_nlu_index_from_apollo(
            config, "nlu_index", "spring.elasticsearch.rest.password"
        )

    def train(self):
        sen_list, labels = self.data[0], self.data[1]
        
        num_sen = len(sen_list)
        print("训练集句子数: ", num_sen)    # train_noise.csv: 173802 → 16119
        K1 = int(num_sen * 0.8)
        Q1 = num_sen - K1
        print("support set: ", K1, "\tquery set: ", Q1)  # 139041 34761

        # # 统计样本数为 x 的类别数 y
        # from collections import Counter
        # print("labels: ", len(labels))
        # c = Counter(labels)
        # c_dict = {}
        # for k,v in dict(c).items():
        #     c_dict.setdefault(v, 0)
        #     c_dict[v] += 1
        # print(sorted(c_dict.items(), key=lambda x: x[1], reverse=True))
        # # (1, 2222), (2, 1371), (21, 930), (3, 493)
        # exit()

        global label2kid
        for kid in labels:
            kid2label.setdefault(kid, len(kid2label))
        label2kid = {v: k for k, v in kid2label.items()}
        num_classes = len(kid2label)
        print("类别数统计:{}".format(num_classes))  #
        # mid = num_sen // num_classes
        # K1, Q1 = mid, None

        # pre-processing training dataset.
        x_train = []
        print("对训练集进行向量映射...")
        print("nlu embedding 接口 API: ", self.es_index_info["spring.embedding.url"])
        for sen in tqdm(sen_list):
            vector = get_nlu_embedding(sen, self.es_index_info["spring.embedding.url"])
            x_train.append(vector)
        x_train = np.array([x_train])

        y = list(map(lambda x: kid2label[x], labels))
        # y_train = np.array([y[-(num_sen - K1 * num_classes) :]])
        y_train = np.array([y[-Q1:]])  # 只需传入 query set 的标签
        y_train = to_categorical(y_train)

        print("x_train.shape: ", x_train.shape)  # (1,num_sen,768)
        print("y_train.shape: ", y_train.shape)  # (1,Q1,num_classes)

        print("========开始训练，学习类表征========")
        t0 = time.time()  # 计算运行时间
        self.induction_nn, self.encoder = self.model.build_model(num_classes, K1, Q1, num_sen)
        callback = CustomCallback(
            self.encoder,   # encoder编码改为请求 nlu_embedding 接口
            x_train,
            self.test_data,
            self.config,
            self.sen2kid,
            label2kid
            # label2vec,
            # label2vec_base,
        )
        self.induction_nn.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            # validation_data=(x_test, y_test)
            callbacks=[callback]
        )
        t1 = time.time()
        print(("\n=========训练耗时：%.2fs========\n" % (t1 - t0)).lstrip("0"))

        exit()

        # 类表示抽取 (待商榷)
        # encoder = Model(inputs=self.xx.input, outputs=self.xx.get_layer("capsule_layer").output)
        class_vector = self.encoder.predict(x_train)  # 存在[nan]的情况，导致无法计算向量相似度
        # print(np.array(class_vector).shape)   # (1, n_class, 768)
        label2vec = {
            label2kid[label]:vec for label, vec in enumerate(class_vector[0])
        }    # { int:[] }

        print("开始评测....")
        c = CustomCallback(
            None,   # encoder编码改为请求 nlu_embedding 接口
            self.test_data,
            self.config,
            self.sen2kid,
            label2vec,
            # label2vec_base,
        )
        print("加归纳层结果：")
        c.on_epoch_end(0)
        # print("\n不加归纳层结果：")
        # c.on_epoch_end2(0)
        # print("\n平均类中心结果：")
        # c.on_epoch_end3(0)


if __name__ == "__main__":

    json_file = "config.json"
    config, _ = get_config_from_json(json_file)

    print("Create the data generator.")
    data_loader = DataLoader(config)
    print("Create the model.")
    model = C_Model(config)
    print("Create the trainer")
    trainer = Trainer(model, data_loader, config)
    print("Start training the model.")
    trainer.train()
    exit()

    # print("++++++++", trainer.model.class_vector)  # must feed a value for input with ...
    # capsule_layer = trainer.xx.get_layer('capsule_layer')
    # print(capsule_layer.get_output_at(node_index=-1))   # 层有多个节点, 使用get_output_at获取想要节点处输出张量的形状
    # # print(K.eval(capsule_layer.get_output_at(node_index=-1)))   #   # 将tensor转化为numpy
    # print(capsule_layer.get_output_shape_at(node_index=-1))   # 层有多个节点, 使用get_output_at获取想要节点处输出数据的形状
    # weights = capsule_layer.get_weights()  # 以Numpy矩阵的形式返回层的权重
    # print("centroids.shape: ", np.array(weights).shape)  # (1, 1, 768, 17664)
    # print(trainer.xx.get_layer('capsule_layer').get_config())

    encoder = Model(
        inputs=trainer.xx.input, outputs=trainer.xx.get_layer("capsule_layer").output
    )
    X_test = get_nlu_embedding(
        "公积金如何办理",
        get_nlu_index_from_apollo(
            config, "nlu_index", "spring.elasticsearch.rest.password"
        )["spring.embedding.url"],
    )  # nlu向量映射
    X_test = np.array([[X_test]])
    print(encoder.predict(X_test).shape)  # (1, 23, 768)

    X_test = get_nlu_embedding(
        "查询我的珑珠余额",
        get_nlu_index_from_apollo(
            config, "nlu_index", "spring.elasticsearch.rest.password"
        )["spring.embedding.url"],
    )  # nlu向量映射
    X_test = np.array([[X_test]])
    print(encoder.predict(X_test).shape)

    Y_pred = trainer.xx.predict(X_test)  # 用模型进行预测
    # Y_pred = Y_pred.argsort()[:,-2:] #取最高分数的两个类别
    Y_pred = np.argmax(Y_pred)
    print(Y_pred)  # 4
    print(label2kid[Y_pred])
