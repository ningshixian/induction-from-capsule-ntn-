#! -*- coding: utf-8 -*-
# refer: https://kexue.fm/archives/5112

from keras import activations
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)  # (1 + s_squared_norm)
    return scale * x


#define our own softmax function instead of K.softmax
#将每个x_i减去x中的最大值再代入以上公式，原因是为了防止上溢和下溢
#https://blog.csdn.net/weixin_38314865/article/details/107568686
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)


#A Capsule Implement with Pure Keras
class Capsule(Layer):
    ''' The routing algorithm.'''
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule  # 类别数
        self.dim_capsule = dim_capsule  # 类别向量维度
        self.routings = routings    # 迭代次数
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        print("input_shape: ", input_shape)  # (None, None, 768)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)    # [1,128,10*16]
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs): # u-[bs,-1,128]
        # conv1d是一维卷积，local_conv1d是不共享权重的一维卷积，它们都是矩阵乘法
        # conv1d(u, w)解释：http://www.4k8k.xyz/article/weixin_43788143/107134977
        #   卷积核 w 在 u 上滑动做矩阵乘法，大小为 (1*128)，每次卷积选1列，每次移动1列，一共 10*16 个 filter 
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)   #[bs,-1,10*16]
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))    #[bs,-1,10,16]
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))    #[bs,10,-1,16] (C,K,H)
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, 1)   # [batch_size, num_capsule, input_num_capsule]
            # o = K.batch_dot(c, u_hat_vecs, [2, 2])
            #   tf.enisum()多维线性代数数组运算 https://icode.best/i/02374131641423
            o = tf.einsum('bin,binj->bij', c, u_hat_vecs)    # o.shape =  [None, num_capsule, dim_capsule]
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)     # [None, 10, 16] 替换 squash()
                # b = K.batch_dot(o, u_hat_vecs, [2, 3])
                b = tf.einsum('bij,binj->bin', o, u_hat_vecs)    #[bs,10,-1]
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        # print(K.eval(b))
        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)   # v-[bs,10,16]
