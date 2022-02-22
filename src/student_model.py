# -*- coding: utf-8 -*-
# Copyright (C) 2018, Tencent Inc.
# Author: bintan

import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tqdm import tqdm



'''
!!!ATTENTION!!!
The following data involves privacy, please set according to your own characteristics
'''

feature_size_dict = {}
context_sparse_feature_list = []
ad_sparse_feature_list = []
context_feature_vicab_dict = {}
context_feature_embedding_dict = {}
ad_feature_vicab_dict = {}
ad_feature_embedding_dict = {}

ENV_AD_SIZE = 1
VISIBLE_FEATURE_DIM = 1
AD_DENSE_FEATRUE_DIM = AD_DENSE_FEATRUE_SIZE = 1
TEACHER_EMBEDDING_DIM = 1

AD_SPARSE_FEATRUE_SIZE = 1
CONTEXT_SPARSE_FEATRUE_SIZE = 1
CONTEXT_DENSE_FEATRUE_SIZE = 1
ENV_SPARSE_FEATRUE_SIZE = AD_SPARSE_FEATRUE_SIZE * ENV_AD_SIZE
ENV_DENSE_FEATRUE_SIZE = AD_DENSE_FEATRUE_SIZE * ENV_AD_SIZE
ENV_EMBEDDING_SIZE = 1
PADDING_MASK_SIZE = ENV_EMBEDDING_SIZE * ENV_AD_SIZE
label_SIZE = 1

L2_REGU = 0.01
VISIBLE_FEATURE_INDEX = VISIBLE_FEATURE_DIM
CONTEXT_DENSE_FEATRUE_INDEX = VISIBLE_FEATURE_INDEX + CONTEXT_SPARSE_FEATRUE_SIZE
AD_SPARSE_FEATRUE_INDEX = CONTEXT_DENSE_FEATRUE_INDEX + CONTEXT_DENSE_FEATRUE_SIZE
AD_DENSE_FEATRUE_INDEX = AD_SPARSE_FEATRUE_INDEX + AD_SPARSE_FEATRUE_SIZE
ENV_SPARSE_FEATRUE_INDEX = AD_DENSE_FEATRUE_INDEX + AD_DENSE_FEATRUE_DIM
ENV_DENSE_FEATRUE_INDEX = ENV_SPARSE_FEATRUE_INDEX + \
    AD_SPARSE_FEATRUE_SIZE * ENV_AD_SIZE
PADDING_MASK_INDEX = ENV_DENSE_FEATRUE_INDEX + ENV_DENSE_FEATRUE_SIZE
label_INDEX = -3


class StudentModel(object):
    def __init__(self, logger):
        self.logger = logger


    def build_teacher_network(self, lr, deep_params, cross_layer_num, concate_dim, with_cross=True):
        self.logger.info("cross_layer_num is : {}".format(cross_layer_num))
        ad_sparse_input = tf.placeholder(tf.int64, [None, AD_SPARSE_FEATRUE_SIZE], name='ad_sparse_input')
        ad_dense_input = tf.placeholder(tf.float32, [None, AD_DENSE_FEATRUE_SIZE], 'ad_dense_input')
        env_ad_sparse_input = tf.placeholder(tf.int64, [None, ENV_AD_SIZE * AD_SPARSE_FEATRUE_SIZE],
                                             name='env_ad_sparse_input')
        env_ad_dense_input = tf.placeholder(tf.float32, [None, ENV_AD_SIZE * AD_DENSE_FEATRUE_SIZE],
                                            'env_ad_dense_input')
        context_sparse_input = tf.placeholder(tf.int64, [None, CONTEXT_SPARSE_FEATRUE_SIZE], 'context_sparse_input')
        context_dense_input = tf.placeholder(tf.float32, [None, CONTEXT_DENSE_FEATRUE_SIZE], 'context_dense_input')
        gd_net_label = tf.placeholder(tf.float32, [None, 1], 'class')
        rtb_net_label = tf.placeholder(tf.float32, [None, 1], 'class')
        teacher_pred_gd = tf.placeholder(tf.float32, [None, 1], 'teacher_pred')
        teacher_embedding_gd = tf.placeholder(tf.float32, [None, 8], 'teacher_embedding')
        teacher_pred_rtb = tf.placeholder(tf.float32, [None, 1], 'teacher_pred')
        teacher_embedding_rtb = tf.placeholder(tf.float32, [None, 8], 'teacher_embedding')
        is_train = tf.placeholder_with_default(False, (), 'is_train')
        # teacher net
        with tf.name_scope("teacher_net"):
            t_embedding_matrix_dict = {}
            t_context_embeddings = []
            t_ad_embeddings = []
            t_env_ad_embeddings = []
            t_cross_layer_weights_dict = {}
            for context_index, key in enumerate(context_sparse_feature_list):
                t_embedding_matrix_dict[key] = tf.Variable(tf.random_normal(
                    shape=[context_feature_vicab_dict[key], context_feature_embedding_dict[key]], mean=0, stddev=1),
                                                           name='{}_embedding_matrix'.format(key))
                t_context_embeddings.append(
                    tf.nn.embedding_lookup(t_embedding_matrix_dict[key], context_sparse_input[:, context_index]))
            t_context_embedding_concated = tf.concat(t_context_embeddings, 1)

            for ad_index, key in enumerate(ad_sparse_feature_list):
                t_embedding_matrix_dict[key] = tf.Variable(tf.random_normal(
                    shape=[ad_feature_vicab_dict[key], ad_feature_embedding_dict[key]], mean=0, stddev=1),
                                                           name='ad_{}_embedding_matrix'.format(key))
                t_ad_embeddings.append(
                    tf.nn.embedding_lookup(t_embedding_matrix_dict[key], ad_sparse_input[:, ad_index]))
            t_ad_embedding_concated = tf.concat(t_ad_embeddings, 1)

            for env_index in range(ENV_AD_SIZE):
                t_env_feature_embeddings = []
                for ad_index, key in enumerate(ad_sparse_feature_list):
                    t_env_feature_embeddings.append(
                        tf.nn.embedding_lookup(t_embedding_matrix_dict[key],
                                               env_ad_sparse_input[:, env_index * AD_SPARSE_FEATRUE_SIZE + ad_index]))
                    t_env_ad_embeddings.append(tf.concat(t_env_feature_embeddings, 1))

            env_ad_weight_list = []
            for ad_index, t_env_ad_embedding in enumerate(t_env_ad_embeddings):
                env_ad_mult = tf.multiply(t_env_ad_embedding, t_ad_embedding_concated)
                env_concat = tf.concat([t_ad_embedding_concated, env_ad_mult, t_env_ad_embedding])
                env_ad_weight_list.append(tf.layers.dense(inputs=env_concat,
                                            units=1,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REGU),
                                            name='env_dense_{}'.format(ad_index)))
            env_ad_weight_tensor = tf.concat(env_ad_weight_list, 1)
            t_env_ad_embedding_weight = tf.matmul(env_ad_weight_tensor, tf.concat(t_env_ad_embeddings, 0)) 
            t_env_ad_embedding_sum_pooling = tf.reduce_sum(t_env_ad_embeddings, axis=0)
            
            t_concate_gd_net = tf.concat([
                t_context_embedding_concated, t_ad_embedding_concated, t_env_ad_embedding_weight, env_ad_dense_input,
                context_dense_input
            ], 1)

            t_concate_rtb_net = tf.concat([
                t_context_embedding_concated, t_ad_embedding_concated, t_env_ad_embedding_sum_pooling, env_ad_dense_input,
                context_dense_input
            ], 1)
            t_concate_gd_net = tf.layers.batch_normalization(inputs=t_concate_gd_net, training=is_train)
            t_concate_rtb_net = tf.layers.batch_normalization(inputs=t_concate_rtb_net, training=is_train)
            # gd_net
            if with_cross:
                for i in range(cross_layer_num):
                    t_cross_layer_weights_dict['cross_layer_weight_{0}'.format(i)] = tf.Variable(
                        tf.random_normal([concate_dim, 1], 0.0, 0.01), tf.float32)
                    t_cross_layer_weights_dict['cross_layer_bias_{0}'.format(i)] = tf.Variable(
                        tf.random_normal([concate_dim, 1], 0.0, 0.01), tf.float32)
                t_cross_input = tf.reshape(t_concate_gd_net, [-1, concate_dim, 1])
                x_now = t_cross_input
                for i in range(cross_layer_num):
                    x_now = tf.add(
                        tf.add(
                            tf.tensordot(tf.matmul(t_cross_input, x_now, transpose_b=True),
                                         t_cross_layer_weights_dict['cross_layer_weight_{0}'.format(i)],
                                         axes=1), t_cross_layer_weights_dict['cross_layer_bias_{0}'.format(i)]), x_now)
                t_cross_output = tf.reshape(x_now, [-1, concate_dim])
                t_fc_data_gd_net = tf.concat([t_cross_output, t_concate_gd_net], 1)
            else:
                t_fc_data_gd_net = t_concate_gd_net
            for i, units in enumerate(deep_params.split(",")):
                t_fc_data_gd_net = tf.layers.dense(inputs=t_fc_data_gd_net,
                                            units=int(units),
                                            activation='relu',
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REGU),
                                            name='dense_{}'.format(i))
                t_fc_data_gd_net = tf.layers.dropout(inputs=t_fc_data_gd_net,
                                              rate=0.5,
                                              training=is_train,
                                              name='dropout_{}'.format(i))
            t_class_gd_net_pred = tf.layers.dense(inputs=t_fc_data_gd_net, units=1, name='class_pred_output')

            # rtb_net
            if with_cross:
                for i in range(cross_layer_num):
                    t_cross_layer_weights_dict['cross_layer_weight_{0}'.format(i)] = tf.Variable(
                        tf.random_normal([concate_dim, 1], 0.0, 0.01), tf.float32)
                    t_cross_layer_weights_dict['cross_layer_bias_{0}'.format(i)] = tf.Variable(
                        tf.random_normal([concate_dim, 1], 0.0, 0.01), tf.float32)
                t_cross_input = tf.reshape(t_concate_rtb_net, [-1, concate_dim, 1])
                x_now = t_cross_input
                for i in range(cross_layer_num):
                    x_now = tf.add(
                        tf.add(
                            tf.tensordot(tf.matmul(t_cross_input, x_now, transpose_b=True),
                                         t_cross_layer_weights_dict['cross_layer_weight_{0}'.format(i)],
                                         axes=1), t_cross_layer_weights_dict['cross_layer_bias_{0}'.format(i)]), x_now)
                t_cross_output = tf.reshape(x_now, [-1, concate_dim])
                t_fc_data_rtb_net = tf.concat([t_cross_output, t_concate_rtb_net], 1)
            else:
                t_fc_data_rtb_net = t_concate_rtb_net
            for i, units in enumerate(deep_params.split(",")):
                t_fc_data_rtb_net = tf.layers.dense(inputs=t_fc_data_rtb_net,
                                            units=int(units),
                                            activation='relu',
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REGU),
                                            name='dense_{}'.format(i))
                t_fc_data_rtb_net = tf.layers.dropout(inputs=t_fc_data_rtb_net,
                                              rate=0.5,
                                              training=is_train,
                                              name='dropout_{}'.format(i))
            t_class_rtb_net_pred = tf.layers.dense(inputs=t_fc_data_rtb_net, units=1, name='class_pred_output')

        gd_soft_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(teacher_pred_gd, t_class_gd_net_pred))
        gd_hard_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(gd_net_label, t_class_gd_net_pred))
        gd_hint_loss = tf.reduce_mean((t_fc_data_rtb_net - teacher_embedding_gd)**2)

        gd_s_fc_data_similar_matrix = tf.matmul(t_fc_data_rtb_net, t_fc_data_rtb_net, transpose_b=True)
        gd_teacher_embedding_similar_matrix = tf.matmul(teacher_embedding_gd, teacher_embedding_gd, transpose_b=True)
        gd_similar_loss = tf.reduce_mean((gd_s_fc_data_similar_matrix - gd_teacher_embedding_similar_matrix)**2)
        gd_class_loss = gd_hard_loss + 0.1 * gd_soft_loss + +1e-7 * gd_hint_loss + 1e-13 * gd_similar_loss

        rtb_soft_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(teacher_pred_rtb, t_class_rtb_net_pred))
        rtb_hard_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(rtb_net_label, t_class_rtb_net_pred))
        rtb_hint_loss = tf.reduce_mean((t_fc_data_rtb_net - teacher_embedding_rtb)**2)

        rtb_s_fc_data_similar_matrix = tf.matmul(t_fc_data_rtb_net, t_fc_data_rtb_net, transpose_b=True)
        rtb_teacher_embedding_similar_matrix = tf.matmul(teacher_embedding_rtb, teacher_embedding_rtb, transpose_b=True)
        rtb_similar_loss = tf.reduce_mean((rtb_s_fc_data_similar_matrix - rtb_teacher_embedding_similar_matrix)**2)
        rtb_class_loss = rtb_hard_loss + 0.1 * rtb_soft_loss + +1e-7 * rtb_hint_loss + 1e-13 * rtb_similar_loss

        loss = gd_class_loss + rtb_class_loss
        t_gd_net_pred_sigmoid = tf.sigmoid(t_class_gd_net_pred, name='class_gd_net_pred_sigmoid')
        t_rtb_net_pred_sigmoid = tf.sigmoid(t_class_rtb_net_pred, name='class_rtb_net_pred_sigmoid')

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            trian_op = tf.train.AdamOptimizer(lr).minimize(loss)

        inputs = [
            ad_sparse_input, ad_dense_input, env_ad_sparse_input, env_ad_dense_input, context_sparse_input,
            context_dense_input, teacher_pred_gd, teacher_embedding_gd, teacher_pred_rtb, teacher_embedding_rtb, 
            gd_net_label, rtb_net_label, is_train
        ]
        return [inputs, trian_op, loss, t_gd_net_pred_sigmoid, t_rtb_net_pred_sigmoid]


    def train(self, samples, model_op, epoches, early_stop, teacher_res_path, save_model, save_dir_name):
        tf.reset_default_graph()
        tf.set_random_seed(1)
        train_samples, valid_samples = samples
        inputs, trian_op, loss, gd_net_pred, rtb_net_pred, output_t_fc_data_gd_net, output_t_fc_data_rtb_net = model_op
        ad_sparse_input, ad_dense_input, env_ad_sparse_input, env_ad_dense_input, context_sparse_input, \
            context_dense_input, teacher_pred_gd, teacher_embedding_gd, teacher_pred_rtb, teacher_embedding_rtb, \
            gd_net_label, rtb_net_label, is_train = inputs
        early_stop_cnt = 0
        last_loss = 0
        self.logger.info("train batches = {}, valid batches = {},".format(len(train_samples), len(valid_samples)))
        teacher_dict = self.utils.load_pickle_file(teacher_res_path)
        with tf.Session() as sess:
            self.logger.info("global_variables_initializer.")
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            for i, epoch in enumerate(range(epoches)):
                train_loss = 0
                train_gd_net_auc = 0
                train_rtb_net_auc = 0
                for batch_sample in train_samples:
                    train_visible_features, train_context_sparse_features, train_context_dense_features, train_ad_sparse_features, \
                        train_ad_dense_features, train_env_sparse_features, train_env_dense_features, train_labels = batch_sample
                    train_labels = train_labels.astype(float)
                    train_gd_net_label = train_labels[:, 0].reshape(-1, 1)
                    train_rtb_net_label = train_labels[:, 1].reshape(-1, 1)
                    gd_train_teacher_pred = []
                    gd_train_teacher_embedding = []
                    rtb_train_teacher_pred = []
                    rtb_train_teacher_embedding = []
                    for row in train_visible_features:
                        req_id, aid = row[:2]
                        pred_gd, embedding_gd, pred_rtb, embedding_rtb = teacher_dict["{}_{}".format(req_id, aid)]
                        gd_train_teacher_pred.append(pred_gd)
                        gd_train_teacher_embedding.append(embedding_gd)
                        rtb_train_teacher_pred.append(pred_rtb)
                        rtb_train_teacher_embedding.append(embedding_rtb)

                    gd_train_teacher_pred_np = np.array(gd_train_teacher_pred).reshape(-1, 1).astype(float)
                    gd_train_teacher_embedding_np = np.array(gd_train_teacher_embedding).reshape(
                        -1, TEACHER_EMBEDDING_DIM).astype(float)
                    rtb_train_teacher_pred_np = np.array(gd_train_teacher_pred).reshape(-1, 1).astype(float)
                    rtb_train_teacher_embedding_np = np.array(gd_train_teacher_embedding).reshape(
                        -1, TEACHER_EMBEDDING_DIM).astype(float)
                    _, train_loss, train_gd_net_pred, train_rtb_net_pred = sess.run(
                        [trian_op, loss, gd_net_pred, rtb_net_pred],
                        feed_dict={
                            context_sparse_input: train_context_sparse_features,
                            context_dense_input: train_context_dense_features,
                            ad_sparse_input: train_ad_sparse_features,
                            ad_dense_input: train_ad_dense_features,
                            env_ad_sparse_input: train_env_sparse_features,
                            env_ad_dense_input: train_env_dense_features,
                            teacher_pred_gd: gd_train_teacher_pred_np,
                            teacher_embedding_gd: gd_train_teacher_embedding_np,
                            teacher_pred_rtb: rtb_train_teacher_pred_np,
                            teacher_embedding_rtb: rtb_train_teacher_embedding_np,
                            gd_net_label: train_gd_net_label,
                            rtb_net_label: train_rtb_net_label,
                            is_train: True
                        })
                    fpr, tpr, _ = metrics.roc_curve(train_gd_net_label, train_gd_net_pred, pos_label=1)
                    train_gd_net_auc += metrics.auc(fpr, tpr)
                    fpr, tpr, _ = metrics.roc_curve(train_rtb_net_label, train_rtb_net_pred, pos_label=1)
                    train_rtb_net_auc += metrics.auc(fpr, tpr)
                    train_loss += train_loss
                valid_loss = 0
                valid_gd_net_auc = 0
                valid_rtb_net_auc = 0
                for batch_sample in valid_samples:
                    valid_visible_features, valid_context_sparse_features, valid_context_dense_features, valid_ad_sparse_features, \
                        valid_ad_dense_features, valid_env_sparse_features, valid_env_dense_features, valid_labels = batch_sample
                    valid_labels = valid_labels.astype(float)
                    valid_gd_net_label = valid_labels[:, 0].reshape(-1, 1)
                    valid_rtb_net_label = valid_labels[:, 1].reshape(-1, 1)
                    gd_valid_teacher_pred = []
                    gd_valid_teacher_embedding = []
                    rtb_valid_teacher_pred = []
                    rtb_valid_teacher_embedding = []
                    for row in valid_visible_features:
                        req_id, aid = row[:2]
                        pred_gd, embedding_gd, pred_rtb, embedding_rtb = teacher_dict["{}_{}".format(req_id, aid)]
                        gd_valid_teacher_pred.append(pred_gd)
                        gd_valid_teacher_embedding.append(embedding_gd)
                        rtb_valid_teacher_pred.append(pred_rtb)
                        rtb_valid_teacher_embedding.append(embedding_rtb)

                    gd_valid_teacher_pred_np = np.array(gd_valid_teacher_pred).reshape(-1, 1).astype(float)
                    gd_valid_teacher_embedding_np = np.array(gd_valid_teacher_embedding).reshape(
                        -1, TEACHER_EMBEDDING_DIM).astype(float)
                    rtb_valid_teacher_pred_np = np.array(gd_valid_teacher_pred).reshape(-1, 1).astype(float)
                    rtb_valid_teacher_embedding_np = np.array(gd_valid_teacher_embedding).reshape(
                        -1, TEACHER_EMBEDDING_DIM).astype(float)
                    valid_loss, valid_gd_net_pred, valid_rtb_net_pred = sess.run(
                        [loss, gd_net_pred, rtb_net_pred],
                        feed_dict={
                            context_sparse_input: valid_context_sparse_features,
                            context_dense_input: valid_context_dense_features,
                            ad_sparse_input: valid_ad_sparse_features,
                            ad_dense_input: valid_ad_dense_features,
                            env_ad_sparse_input: valid_env_sparse_features,
                            env_ad_dense_input: valid_env_dense_features,
                            teacher_pred_gd: gd_valid_teacher_pred_np,
                            teacher_embedding_gd: gd_valid_teacher_embedding_np,
                            teacher_pred_rtb: rtb_valid_teacher_pred_np,
                            teacher_embedding_rtb: rtb_valid_teacher_embedding_np,
                            gd_net_label: valid_gd_net_label,
                            rtb_net_label: valid_rtb_net_label,
                            is_train: True
                        })
                    fpr, tpr, _ = metrics.roc_curve(valid_gd_net_label, valid_gd_net_pred, pos_label=1)
                    valid_gd_net_auc += metrics.auc(fpr, tpr)
                    fpr, tpr, _ = metrics.roc_curve(valid_rtb_net_label, valid_rtb_net_pred, pos_label=1)
                    valid_rtb_net_auc += metrics.auc(fpr, tpr)
                    valid_loss += valid_loss

                train_loss /= len(train_samples)
                valid_loss /= len(valid_samples)
                train_gd_net_auc /= len(train_samples)
                train_rtb_net_auc /= len(train_samples)
                valid_gd_net_auc /= len(valid_samples)
                valid_rtb_net_auc /= len(valid_samples)

                metric_str = "epoch {}, loss = {:.6f}, valid loss = {:.6f}, train_gd_auc = {:.6f}, valid_gd_auc = {:.6f}, train_rtb_auc = {:.6f}, valid_rtb_auc = {:.6f},early_stop_cnt = {}".format(
                    epoch, train_loss, valid_loss, train_gd_net_auc, valid_gd_net_auc, train_rtb_net_auc, valid_rtb_net_auc, early_stop_cnt)
                self.logger.info(metric_str)
                
                if save_model and valid_loss < last_loss:
                    builder = tf.saved_model.builder.SavedModelBuilder(save_dir_name)
                    tensor_info_context_sparse_input = tf.saved_model.utils.build_tensor_info(context_sparse_input)
                    tensor_info_context_dense_input = tf.saved_model.utils.build_tensor_info(context_dense_input)
                    tensor_info_ad_sparse_input = tf.saved_model.utils.build_tensor_info(ad_sparse_input)
                    tensor_info_ad_dense_input = tf.saved_model.utils.build_tensor_info(ad_dense_input)
                    tensor_info_env_ad_sparse_input = tf.saved_model.utils.build_tensor_info(env_ad_sparse_input)
                    tensor_info_env_ad_dense_input = tf.saved_model.utils.build_tensor_info(env_ad_dense_input)
                    tensor_info_gd_net_label = tf.saved_model.utils.build_tensor_info(gd_net_label)
                    tensor_info_rtb_net_label = tf.saved_model.utils.build_tensor_info(rtb_net_label)
                    tensor_info_output_t_fc_data_gd_net = tf.saved_model.utils.build_tensor_info(output_t_fc_data_gd_net)
                    tensor_info_output_t_fc_data_rtb_net = tf.saved_model.utils.build_tensor_info(output_t_fc_data_rtb_net)
                    tensor_info_gd_net_pred = tf.saved_model.utils.build_tensor_info(gd_net_pred)
                    tensor_info_rtb_net_pred = tf.saved_model.utils.build_tensor_info(rtb_net_pred)
                    tensor_info_loss = tf.saved_model.utils.build_tensor_info(loss)
                    prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
                        inputs={
                            'context_sparse_input': tensor_info_context_sparse_input,
                            'context_dense_input': tensor_info_context_dense_input,
                            'ad_sparse_input': tensor_info_ad_sparse_input,
                            'ad_dense_input': tensor_info_ad_dense_input,
                            'env_ad_sparse_input': tensor_info_env_ad_sparse_input,
                            'env_ad_dense_input': tensor_info_env_ad_dense_input,
                            'gd_net_label': tensor_info_gd_net_label,
                            'rtb_net_label': tensor_info_rtb_net_label
                        },
                        outputs={
                            'output_t_fc_data_gd_net': tensor_info_output_t_fc_data_gd_net,
                            'output_t_fc_data_rtb_net': tensor_info_output_t_fc_data_rtb_net,
                            'gd_net_pred': tensor_info_gd_net_pred,
                            'rtb_net_pred': tensor_info_rtb_net_pred,
                            'loss': tensor_info_loss
                        },
                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
                    builder.add_meta_graph_and_variables(
                        sess,
                        [tf.saved_model.tag_constants.SERVING],
                        signature_def_map={
                            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
                        },
                    )
                    builder.save()
                if i != 0:
                    if valid_loss >= last_loss:
                        early_stop_cnt += 1
                    else:
                        early_stop_cnt = 0
                last_loss = valid_loss
                if early_stop_cnt >= early_stop:
                    break
