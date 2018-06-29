#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

__author__ = 'luheng'

import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from batch_data import generate_batch
from batch_data import generate_batch_shell
from batch_data import generate_batch_evaluate
from batch_data import embeddings
from datetime import datetime
from util import Progbar
import sys
from sklearn import metrics
import os
import time
from parameters import Para
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

logger = logging.getLogger("little_try")
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
fh = logging.FileHandler('result.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

vec, voc = embeddings()
embedding_size = Para.embedding_size
vocab_size = len(vec)
batch_size = Para.batch_size
num_units = Para.num_units
max_gradient_norm = Para.max_gradient_norm
learning_rate = Para.learning_rate
n_epochs = Para.n_epochs
n_outputs = Para.n_outputs
train_keep_prob = Para.train_keep_prob
train_num = Para.train_num
dev_num = Para.dev_num
test_num = Para.test_num
l2_rate = Para.l2_rate
best_loss = np.infty
threshold = Para.threshold
max_sentence = Para.max_sentence
filter_sizes = Para.filter_sizes
epochs_without_progress = 0
max_epochs_without_progress = 50
l2_loss = tf.constant(0.0)
checkpoint_path = "model/tmp/my_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "model/my_logreg_model"
label_dict = {'0': '无意义回复', '1': "我不需要|地点不符合|利息太高", '2': "你发信息|加微信|QQ|回头打电话|我在开车|我在忙", '3': "询问:能贷多少|额度", '4': "询问:利息",
              '5': "询问:什么公司|平台|银行", '6': "询问:下款时间", '7': "询问:收费|服务费", '8': "询问:公司地点", '9': "询问:没听清|再说一遍", '10': "询问:贷款手续|所需文件|要求|我能贷款吗",
              '11': "询问:你是机器人吗", '12': "询问:你怎么知道我号码的", '13': "我们是同行", '14': "肯定|接着说"}


'''
tricks:
1,he_init or Xavier or None
2,GRU or Bi-GRU
3,LuongAttention or BahdanauAttention
4,L1_regularization or L2_regularization or None
5,Gradient Clipping or None
6,Adam Momentum RMSProp Adadelta
'''
with tf.name_scope("placeholder"):
    encode_question = tf.placeholder(tf.int32, shape=[batch_size, None])
    encode_answer = tf.placeholder(tf.int32, shape=[batch_size, None])
    question  = encode_question
    answer = encode_answer
    simi_values = tf.placeholder(tf.int32, shape=[batch_size])
    keep_prob = tf.placeholder_with_default(1.0, shape=())

with tf.name_scope('word_embeddings'):
    embeddings = tf.Variable(vec, trainable=True, name="embeds")
    question_emb = tf.nn.embedding_lookup(embeddings, question)
    answer_emb = tf.nn.embedding_lookup(embeddings, answer)
    answer_emb_expand = tf.expand_dims(answer_emb, -1)

with tf.name_scope("decode"):
    # initializer
    he_init = tc.layers.variance_scaling_initializer()
    Xavier = tc.layers.xavier_initializer()

    '''
    text-cnn
    '''
    with tf.variable_scope("answer"):
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                conv2d = tf.layers.conv2d(answer_emb_expand,filters=2,kernel_size=(filter_size,embedding_size))
                # pool = tf.nn.max_pool(conv2d, ksize=[1, max_sentence - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
                pool = tf.reduce_max(conv2d,1,keepdims=True)
                pooled_outputs.append(pool)

        # Combine all the pooled features
        num_filters_total = 2 * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool = tf.reshape(h_pool, [-1, num_filters_total]) 
        # h_pool = tf.nn.dropout(h_pool, train_keep_prob)             
        fc = tf.layers.dense(h_pool, num_units, name="dense")
        fc = tf.nn.dropout(fc, train_keep_prob) 
        fc = tf.nn.relu(fc)

with tf.name_scope("logits"):
    logits = tf.layers.dense(fc, n_outputs, name="dense")
    labels = simi_values
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    base_loss = tf.reduce_mean(xentropy)
    # regularization
    reg_losses = tc.layers.apply_regularization(
        tc.layers.l2_regularizer(l2_rate), tf.trainable_variables())
    loss = base_loss + reg_losses
    loss_summary = tf.summary.scalar('log_loss', loss)

with tf.name_scope("optimizer"):
    # optimizer = tf.train.MomentumOptimizer(momentum=0.9,learning_rate=learning_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # training_op = optimizer.minimize(loss)
    # Gradient Clipping
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
                  for grad, var in grads_and_vars]
    training_op = optimizer.apply_gradients(capped_gvs)
    pred = tf.argmax(logits, axis=-1)
    correct = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('log_accuracy', accuracy)


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "model/tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


logdir = log_dir("logreg")

init = tf.global_variables_initializer()
saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


def do_train(args):
    tic = time.time()
    train_data = args.train_data
    dev_data = args.dev_data
    epochs_without_progress = 0
    max_epochs_without_progress = 50
    best_f1 = 0.0
    best_epoch = 0
    best_precision = 0.0
    best_recall = 0.0
    with tf.Session(config=config) as sess:
        train_gen = generate_batch(
            train_data, batch_size=batch_size)
        dev_gen = generate_batch(dev_data, batch_size=batch_size)
        if os.path.isfile(checkpoint_epoch_path):
            with open(checkpoint_epoch_path, "rb") as f:
                start_epoch = int(f.read())
            logger.info(
                "Training was interrupted. Continuing at epoch：{}".format(start_epoch))
            saver.restore(sess, checkpoint_path)
        else:
            start_epoch = 0
            sess.run(init)
        for epoch in range(n_epochs):
            prog = Progbar(target=1 + int(train_num / batch_size))
            for i in range(int(train_num / batch_size)):
                source_batch_pad, target_batch_pad, simi_input_batch = next(
                    train_gen)
                pred_, logits_, loss_, loss_summary_, _ = sess.run([pred, logits, loss, loss_summary, training_op], feed_dict={
                    encode_question: source_batch_pad, encode_answer: target_batch_pad,
                    simi_values: simi_input_batch, keep_prob: train_keep_prob})
                y_true = np.array(simi_input_batch)
                precision = metrics.precision_score(
                    y_true, pred_, average='micro')
                recall = metrics.recall_score(y_true, pred_, average='micro')
                f1 = metrics.f1_score(y_true, pred_, average='micro')
                if i % 10 == 0:
                    file_writer.add_summary(
                        loss_summary_, epoch * train_num + i)
                prog.update(i + 1, [("train loss", loss_), ("precision",
                                                            precision), ("recall", recall), ("f1", f1)])
            print('\nresult of this train epoch:')
            logger.info("epoch:{}".format(epoch) +
                        "\tepoch_loss:{:.5f}".format(loss_))
            if epoch % 1 == 0:
                num = int(dev_num / batch_size)
                # dev_accuracy = 0.0
                dev_loss = 0.0
                y_true = []
                y_pred = []
                for i in range(num):
                    dev1_pad, dev2_pad, dev_simi_batch = next(
                        dev_gen)
                    pred_dev, loss_dev = sess.run([pred, loss], feed_dict={
                        encode_question: dev1_pad, encode_answer: dev2_pad,
                        simi_values: dev_simi_batch})
                    y_true = np.append(y_true, dev_simi_batch)
                    y_pred = np.append(y_pred, pred_dev)
                    dev_loss += loss_dev
                y_true = np.array(y_true)
                precision_dev = metrics.precision_score(
                    y_true, y_pred, average='micro')
                recall_dev = metrics.recall_score(
                    y_true, y_pred, average='micro')
                f1_dev = metrics.f1_score(y_true, y_pred, average='micro')
                logger.info("epoch:{}".format(epoch)+"\tDev loss:{:.5f}".format(dev_loss / num)+"\tprecision_dev:{:.5f}".format(
                    precision_dev)+"\trecall_dev:{:.5f}".format(recall_dev)+"\tf1_dev:{:.5f}".format(f1_dev))
                saver.save(sess, checkpoint_path)
                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))
                if f1_dev > best_f1:
                    logger.info('save epoch:{}'.format(epoch))
                    saver.save(sess, final_model_path)
                    best_f1 = f1_dev
                    best_precision = precision_dev
                    best_epoch = epoch
                    best_recall = recall_dev
                else:
                    epochs_without_progress += 1
                    if epochs_without_progress > max_epochs_without_progress:
                        logger.info("Early stopping")
                        break
        os.remove(checkpoint_epoch_path)
        tok = time.time()
        cost = tok-tic
        logger.info("best_epoch:{}".format(best_epoch)+"\tbest_precision:{:.5f}".format(
                    best_precision)+"\tbest_recall:{:.5f}".format(best_recall)+"\tbest_f1:{:.5f}".format(best_f1))
        logger.info('final training time:{:.2f}'.format(cost))


def do_evaluate(args):
    tic = time.time()
    inpath = args.test_data
    outpath = args.output_data
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        test_gen = generate_batch_evaluate(inpath)
        with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
            for _ in fin:
                ids, source, target = next(
                    test_gen)
                preds = sess.run(pred, feed_dict={encode_question: source, encode_answer: target,})
                predict = preds[0]
                fout.write(ids[0] + '\t' + str(predict) + '\n')
    logger.info('evaluation done! out_path:{}'.format(outpath))
    tok = time.time()
    cost = tok-tic
    single_cost = cost*1000/(test_num)
    logger.info(
        'each evaluate time of single data:{:.2f}ms'.format(single_cost))
# # tensorboard --logdir=tf_logs


def do_shell(_):
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        sentence = '请输入句子,要退出请输入：bye'
        logger.info(sentence)
        while sentence != 'bye':
            sentence = input("input> ")
            if sentence == 'bye':
                logger.info("准备退出...")
            elif sentence == '':
                logger.info("请不要打回车玩哦！")
            elif '\t' not in sentence:
                logger.info("请输入正确的格式！提示：Question\tAnswer;退出请输入：bye")
            else:
                shell_gen = generate_batch_shell(sentence)
                source, target = next(shell_gen)
                preds = sess.run(pred, feed_dict={encode_question: source, encode_answer: target})
                predict = preds[0]
                intention = label_dict[str(predict)]
                logger.info("输入原句：{}".format(sentence))
                logger.info("输出意图：{}".format(intention))
        logger.info('谢谢使用，再见！')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains and tests an classification model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help=' ')
    command_parser.add_argument(
        '-td', '--train-data', type=str, default="data/train.csv", help="Training data")
    command_parser.add_argument(
        '-dd', '--dev-data', type=str, default="data/dev.csv", help="Dev data")
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help=' ')
    command_parser.add_argument(
        '-t', '--test-data', type=str, default="data/test.csv", help="Evaluate data")
    command_parser.add_argument(
        '-o', '--output-data', type=str, default="data/out.csv", help="Output data")
    command_parser.set_defaults(func=do_evaluate)

    command_parser = subparsers.add_parser('test', help=' ')
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
