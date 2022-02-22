# -*- coding: utf-8 -*-
# Copyright (C) 2018, Tencent Inc.
# Author: bintan

from __future__ import absolute_import, division, print_function
import time
import numpy as np
import tensorflow as tf
from io import BytesIO
from numpy.lib import format
from operator import add
from pyspark.sql import SparkSession


def generator_ij(iterator):
    buf = np.zeros(shape=(0, 3)).astype('int64')
    for line in iterator:
        biparite_line = line.split('\t')
        orders = biparite_line[4].split(',')
        tmp = np.zeros(shape=(len(orders), 3)).astype('int64')
        # j,i,k
        tmp[:, 0] = orders
        tmp[:, 1] = biparite_line[1]
        tmp[:, 2] = biparite_line[0]

        buf = np.row_stack((buf, tmp))
    yield buf


def init_sp(iterator):

    tf.enable_eager_execution()
    j_nums = bc_j.value
    for e in iterator:
        # print("e is {}".format(e))
        j, i, k = 0, 1, 2
        # es = e.shape[0]
        mask_j = np.zeros(j_nums)
        mask_id = np.random.randint(j_nums, size=500)
        for x in mask_id:
            mask_j[x] = 1
        dense_shape = [j_nums, e[-1][i], e[-1][k]]
        indices = tf.constant(e - [1, 1, 1], dtype=tf.int64)
        indices = tf.boolean_mask(indices, tf.gather(mask_j, indices[:, j]))
        sp_ji = tf.sparse.SparseTensor(indices[:, :2], tf.fill([indices.shape[0]], 1.0), dense_shape[:2])
        sub_sj = tf.sparse.reduce_sum(sp_ji, axis=i).numpy()
        yield [indices, dense_shape, sub_sj]


def get_tbl_key(id_j, id_u):
    key_high = tf.bitwise.left_shift(id_j, 32)
    return tf.bitwise.bitwise_or(key_high, id_u)


def opt_lambda(sum_i, f_j, lambda_0, lambda_1, lambda_):
    lambda_0 = tf.where(sum_i < f_j, lambda_0, lambda_)
    lambda_1 = tf.where(sum_i < f_j, lambda_, lambda_1)
    lambda_ = (lambda_0 + lambda_1) / 2
    return lambda_0, lambda_1, lambda_


def init_loop(iterator):
    tf.enable_eager_execution()
    theta_j = bc_theta.value
    freq_j = bc_freq.value
    max_beta_ = bc_max_beta.value
    indices, dense_shape = 0, 1
    j, i, u = 0, 1, 2
    for x in iterator:
        gather_theta = tf.gather(theta_j, x[indices][:, j])
        xij_clip = tf.clip_by_value(gather_theta, 0, 1)
        xij = tf.sparse.SparseTensor(indices=x[indices], values=xij_clip, dense_shape=x[dense_shape])

        sum_iu = tf.sparse.reduce_sum(xij, axis=[i, u]).numpy()

        sum_i = tf.sparse.reduce_sum_sparse(xij, axis=i)
        tbl_key = get_tbl_key(sum_i.indices[:, 0], sum_i.indices[:, 1])  # id_j, id_u
        tbl_init = tf.contrib.lookup.KeyValueTensorInitializer(keys=tbl_key, values=tf.range(0, tbl_key.shape[0]))
        tbl = tf.contrib.lookup.HashTable(initializer=tbl_init, default_value=-1)
        xij_key = get_tbl_key(xij.indices[:, j], xij.indices[:, u])
        lambda_idx = tbl.lookup(xij_key)

        lambda_ = tf.zeros(sum_i.values.shape)
        lambda_0 = tf.zeros(lambda_.shape)
        lambda_1 = tf.fill(lambda_.shape, -max_beta_)
        gather_freq = tf.gather(freq_j, sum_i.indices[:, j])  # sum_i--->id_j
        lambda_0, lambda_1, lambda_ = opt_lambda(sum_i.values, gather_freq, lambda_0, lambda_1, lambda_)

        b = tf.zeros(x[dense_shape][i])
        b0 = tf.zeros(b.shape)
        b1 = tf.fill(b.shape, -max_beta_)
        sum_ju = tf.sparse.reduce_sum(xij, axis=[j, u])
        b0, b1, b = opt_lambda(sum_ju, 1, b0, b1, b)
        yield [x[indices], x[dense_shape], lambda_idx, gather_theta, gather_freq, b0, b1, b, lambda_0, lambda_1,
               lambda_, 0, sum_iu]


def main_loop(iterator):
    tf.enable_eager_execution()
    alpha_j = bc_alpha.value + 1
    indices, dense_shape, lambda_idx, theta_j, freq_j, b0, b1, b, lambda_0, lambda_1, lambda_ = [i for i in range(11)]
    j, i, u = 0, 1, 2
    for x in iterator:
        spv_alpha = tf.gather(alpha_j, x[indices][:, j])
        spv_beta = tf.gather(x[b], x[indices][:, i])
        spv_lambda = tf.gather(x[lambda_], lambda_idx)
        sum_dual = spv_alpha + spv_beta + spv_lambda
        xij_clip = tf.clip_by_value(sum_dual * x[theta_j], 0, 1)
        xij = tf.SparseTensor(x[indices], xij_clip, x[dense_shape])
        sum_iu = tf.sparse.reduce_sum(xij, axis=[i, u]).numpy()  # 44.2%

        sum_i = tf.sparse.reduce_sum_sparse(xij, axis=i)  # 45.2%
        x[lambda_0], x[lambda_1], x[lambda_] = opt_lambda(sum_i.values, x[freq_j], x[lambda_0], x[lambda_1], x[lambda_])
        sum_ju = tf.sparse.reduce_sum(xij, axis=[j, u])  # 10.6%
        x[b0], x[b1], x[b] = opt_lambda(sum_ju, 1, x[b0], x[b1], x[b])
        x[-2] = xij.values
        x[-1] = sum_iu
        yield x



def opt_alpha(s_j, a0, a1, a):
    a0 = np.where(s_j > d, a0, a)
    a1 = np.where(s_j > d, a, a1)
    a = (a0 + a1) / 2
    return a0, a1, a


def flat(indices, xij):
    for x, y in zip(indices[:, :2].numpy(), xij.numpy()):
        yield '\t'.join(['\t'.join('%d' % i for i in x), str(y)])


if __name__ == "__main__":
    import sys
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    biparitePath = sys.argv[0]
    ordersPath = sys.argv[1]
    partitions = int(sys.argv[2])
    print('biparitePath:' + biparitePath)
    print('ordersPath:' + ordersPath)
    orderDF = spark.read.json(ordersPath).filter("order_play_type = 'out'") \
        .select("vid", "should_play", "freq_times") \
        .collect()

    orderDF = pd.DataFrame(orderDF, columns=["vid", "should_play", "freq_times"])
    orderDF['should_play'].fillna(0, inplace=True)

    d = (orderDF['should_play'] * 1000).astype(np.float32)
    f = orderDF['freq_times'].values.astype(np.float32)

    j = d.shape[0]
    bc_j = sc.broadcast(j)
    bc_freq = sc.broadcast(f)
    init = sc.textFile(biparitePath, partitions).mapPartitions(generator_ij).mapPartitions(init_sp).setName('init_sp').cache()
    sj = init.map(lambda x: x[-1]).treeReduce(add)
    theta = np.where(sj == 0, 0, d / sj).astype(np.float32)

    bc_theta = sc.broadcast(theta)

    max_alpha = 10000.0
    max_beta = max_alpha + 10
    bc_max_beta = sc.broadcast(max_beta)
    alpha = np.zeros(j, dtype=np.float32)
    alpha_0 = np.zeros(j, dtype=np.float32)
    alpha_1 = np.full(j, max_alpha, dtype=np.float32)

    loop = init.mapPartitions(init_loop).setName('loop_original').cache() 
    sum_j = loop.map(lambda x: x[-1]).treeReduce(add)
    init.unpersist()

    alpha_0, alpha_1, alpha = opt_alpha(sum_j, alpha_0, alpha_1, alpha)

    ori_loss = 0
    count = 1
    tc_w = []
    tc_i = []
    while True:
        bc_alpha = sc.broadcast(alpha)
        start = time.time()
        new_loop = loop.mapPartitions(main_loop).setName('loop_%d' % count).cache()
        new_loop.count()
        print('new_loop: ' + str(time.time() - start))
        sum_j = new_loop.map(lambda x: x[-1]).treeReduce(add)
        loop.unpersist()
        loop = new_loop
        alpha_0, alpha_1, alpha = opt_alpha(sum_j, alpha_0, alpha_1, alpha)
        print('opt_alpha: ' + str(time.time() - start))
        count += 1
        if count > 20:
            break

