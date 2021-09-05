# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import custom_optimization
import tensorflow as tf
import numpy as np
import sys
import pickle

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_float("alpha", 0.5, "The weight of loss.")
flags.DEFINE_float("beta", 0.02, "The truncation.")
flags.DEFINE_float("gamma", 2, "The weight of knowledge.")


flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "train_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "test_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "checkpointDir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("signature", 'default', "signature_name")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence. "
                     "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("user_embedding", False, "Whether to add user_embedding.")

#flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("batch_size", 32, "Total batch size for training.")

#flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 1000, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("use_pop_random", False, "use pop random negative samples")
flags.DEFINE_string("vocab_filename", None, "vocab filename")
flags.DEFINE_string("user_history_filename", None, "user history filename")

flags.DEFINE_integer("num_gpu_cores", 2, "num_gpu_cores.")
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,3' #同时使用GPU 0,1,2

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
memory_gpu = ','.join(map(lambda x:str(x),np.argsort(memory_gpu)[::-1][:2]))
os.environ['CUDA_VISIBLE_DEVICES']=memory_gpu
os.system('rm tmp')

class EvalHooks(tf.train.SessionRunHook):
    def __init__(self):
        tf.logging.info('run init')

    def begin(self):
        self.valid_user = 0.0

        self.ndcg_1_A = 0.0
        self.hit_1_A = 0.0
        self.ndcg_5_A = 0.0
        self.hit_5_A = 0.0
        self.ndcg_10_A = 0.0
        self.hit_10_A = 0.0
        self.ap_A = 0.0

        self.ndcg_1_B = 0.0
        self.hit_1_B = 0.0
        self.ndcg_5_B = 0.0
        self.hit_5_B = 0.0
        self.ndcg_10_B = 0.0
        self.hit_10_B = 0.0
        self.ap_B = 0.0

        np.random.seed(12345)

        self.vocab = None

        if FLAGS.user_history_filename is not None:
            print('load user history from :' + FLAGS.user_history_filename)
            with open(FLAGS.user_history_filename, 'rb') as input_file:
                self.user_history = pickle.load(input_file)

        if FLAGS.vocab_filename is not None:
            print('load vocab from :' + FLAGS.vocab_filename)
            with open(FLAGS.vocab_filename, 'rb') as input_file:
                self.vocab = pickle.load(input_file)

            keys = self.vocab.counter.keys()
            values = self.vocab.counter.values()
            self.ids = self.vocab.convert_tokens_to_ids(keys)
            # normalize
            # print(values)
            sum_value = np.sum([x for x in values])
            # print(sum_value)
            self.probability = [value / sum_value for value in values]

    def end(self, session):
        print(
            "\nModel_A: ndcg@1:{}, hit@1:{}, ndcg@5:{}, hit@5:{}, ndcg@10:{}, hit@10:{}, ap:{}, valid_user:{}".
            format(self.ndcg_1_A / self.valid_user, self.hit_1_A / self.valid_user,
                   self.ndcg_5_A / self.valid_user, self.hit_5_A / self.valid_user,
                   self.ndcg_10_A / self.valid_user,
                   self.hit_10_A / self.valid_user, self.ap_A / self.valid_user,
                   self.valid_user))
        print(self.ndcg_1_A, self.hit_1_A, self.ndcg_5_A, self.hit_5_A, self.ndcg_10_A, self.hit_10_A, self.ap_A)

        print(
            "Model_B: ndcg@1:{}, hit@1:{}, ndcg@5:{}, hit@5:{}, ndcg@10:{}, hit@10:{}, ap:{}, valid_user:{}".
                format(self.ndcg_1_B / self.valid_user, self.hit_1_B / self.valid_user,
                       self.ndcg_5_B / self.valid_user, self.hit_5_B / self.valid_user,
                       self.ndcg_10_B / self.valid_user,
                       self.hit_10_B / self.valid_user, self.ap_B / self.valid_user,
                       self.valid_user))
        print(self.ndcg_1_B, self.hit_1_B, self.ndcg_5_B, self.hit_5_B, self.ndcg_10_B, self.hit_10_B, self.ap_B)

    def before_run(self, run_context):
        #tf.logging.info('run before run')
        #print('run before_run')
        variables = tf.get_collection('eval_sp')
        return tf.train.SessionRunArgs(variables)

    def after_run(self, run_context, run_values):
        #tf.logging.info('run after run')
        #print('run after run')
        probs_1, probs_2, input_ids, masked_lm_ids, info = run_values.results
        masked_lm_probs_A = probs_1.reshape(
            (-1, FLAGS.max_predictions_per_seq, probs_1.shape[1]))
        masked_lm_probs_B = probs_2.reshape(
            (-1, FLAGS.max_predictions_per_seq, probs_2.shape[1]))
#         print("loss value:", masked_lm_log_probs.shape, input_ids.shape,
#               masked_lm_ids.shape, info.shape)

        for idx in range(len(input_ids)):
            rated = set(input_ids[idx])
            rated.add(0)
            rated.add(masked_lm_ids[idx][0])
            map(lambda x: rated.add(x),
                self.user_history["user_" + str(info[idx][0])][0])
            item_idx = [masked_lm_ids[idx][0]]
            # here we need more consideration
            masked_lm_probs_elem_A = masked_lm_probs_A[idx, 0]
            masked_lm_probs_elem_B = masked_lm_probs_B[idx, 0]
            size_of_prob = len(self.ids) + 1  # len(masked_lm_log_probs_elem)
            if FLAGS.use_pop_random:
                if self.vocab is not None:
                    while len(item_idx) < 100:
                        sampled_ids = np.random.choice(self.ids, 100, replace=False, p=self.probability)
                        sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
                        item_idx.extend(sampled_ids[:])
                    item_idx = item_idx[:100]
            else:
                # print("evaluation random -> ")
                for _ in range(99):
                    t = np.random.randint(1, size_of_prob)
                    while t in rated:
                        t = np.random.randint(1, size_of_prob)
                    item_idx.append(t)

            predictions_A = -masked_lm_probs_elem_A[item_idx]
            predictions_B = -masked_lm_probs_elem_B[item_idx]
            rank_A = predictions_A.argsort().argsort()[0]
            rank_B = predictions_B.argsort().argsort()[0]

            self.valid_user += 1

            if self.valid_user % 100 == 0:
                print('.', end='')
                sys.stdout.flush()
            
            if rank_A < 1:
                self.ndcg_1_A += 1
                self.hit_1_A += 1
            if rank_A < 5:
                self.ndcg_5_A += 1 / np.log2(rank_A + 2)
                self.hit_5_A += 1
            if rank_A < 10:
                self.ndcg_10_A += 1 / np.log2(rank_A + 2)
                self.hit_10_A += 1

            self.ap_A += 1.0 / (rank_A + 1)
            
            if rank_B < 1:
                self.ndcg_1_B += 1
                self.hit_1_B += 1
            if rank_B < 5:
                self.ndcg_5_B += 1 / np.log2(rank_B + 2)
                self.hit_5_B += 1
            if rank_B < 10:
                self.ndcg_10_B += 1 / np.log2(rank_B + 2)
                self.hit_10_B += 1

            self.ap_B += 1.0 / (rank_B + 1)

            

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, item_size):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name,
                                                         features[name].shape))

        info = features["info"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        embedding = modeling.Embedding(config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            info = info,
            use_one_hot_embeddings=use_one_hot_embeddings)

        model_A = modeling.BertModel(
            config=bert_config,
            input_ids=input_ids,
            is_training=is_training,
            input_tensor=embedding.get_embedding_output(),
            input_mask=input_mask,
            scope='model_A')

        model_B = modeling.BertModel(
            config=bert_config,
            input_ids=input_ids,
            is_training=is_training,
            input_tensor=embedding.get_embedding_output(),
            input_mask=input_mask,
            scope='model_B')

        input_tensor_list = tf.stack([model_A.get_sequence_output(), model_B.get_sequence_output()], 0)

        (loss_A_hard, loss_A_soft, loss_B_hard, loss_B_soft, probs_A, probs_B) = get_masked_lm_output(
            bert_config, input_tensor_list,
            embedding.get_embedding_table(), masked_lm_positions, masked_lm_ids,
            masked_lm_weights)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                 tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint,
                                                  assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            if FLAGS.num_gpu_cores >=2:
                train_op, loss = custom_optimization.create_optimizer(loss_A_hard, loss_A_soft,
                                                               loss_B_hard, loss_B_soft,
                                                               FLAGS.alpha,
                                                               learning_rate,
                                                               num_train_steps,
                                                               num_warmup_steps, use_tpu)
            else:
                train_op, loss = optimization.create_optimizer(loss_A_hard, loss_A_soft,
                                                               loss_B_hard, loss_B_soft,
                                                               FLAGS.alpha,
                                                               learning_rate,
                                                               num_train_steps,
                                                               num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                scaffold=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(loss_A_hard, loss_B_hard, probs_A, probs_B,
                          masked_lm_ids, masked_lm_weights):
                """Computes the loss and accuracy of the model."""
                loss_A = tf.metrics.mean(values=loss_A_hard)
                loss_B = tf.metrics.mean(values=loss_B_hard)
                masked_lm_probs_A = tf.reshape(probs_A, [-1, probs_A.shape[-1]])
                masked_lm_probs_B = tf.reshape(probs_B, [-1, probs_B.shape[-1]])
                masked_lm_predictions_A = tf.argmax(masked_lm_probs_A, axis=-1, output_type=tf.int32)
                masked_lm_predictions_B = tf.argmax(masked_lm_probs_B, axis=-1, output_type=tf.int32)
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy_A = tf.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions_A,
                    weights=masked_lm_weights)
                masked_lm_accuracy_B = tf.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions_B,
                    weights=masked_lm_weights)

                return {
                    "model_A_masked_lm_accuracy": masked_lm_accuracy_A,
                    "model_A_lm_loss": loss_A,
                    "model_B_masked_lm_accuracy": masked_lm_accuracy_B,
                    "model_B_lm_loss": loss_B,
                }

            tf.add_to_collection('eval_sp', probs_A)
            tf.add_to_collection('eval_sp', probs_B)
            tf.add_to_collection('eval_sp', input_ids)
            tf.add_to_collection('eval_sp', masked_lm_ids)
            tf.add_to_collection('eval_sp', info)

            eval_metrics = metric_fn(loss_A_hard, loss_B_hard, probs_A, probs_B,
                                     masked_lm_ids, masked_lm_weights)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss_A_hard+loss_B_hard,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))
        return output_spec
    return model_fn


def get_masked_lm_output(bert_config, input_tensor_list, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    # [model_num*batch_size*label_size, dim]
    sequence_shape = modeling.get_shape_list(input_tensor_list, expected_rank=4)
    model_num = sequence_shape[0]
    gather_list = []
    for i in range(model_num):
        gather_list.append(gather_indexes(input_tensor_list[i], positions))
    input_tensor = tf.concat(gather_list, 0)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[output_weights.shape[0]],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        # logits, (bs*label_size, vocab_size)
        logits = tf.reshape(logits, [model_num, -1, output_weights.shape[0]])

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=output_weights.shape[0], dtype=tf.float32)

        loss_A_hard, loss_A_soft, loss_B_hard, loss_B_soft, probs_A, probs_B = \
            cal_total_loss(logits[0], logits[1], label_weights, one_hot_labels)

    return (loss_A_hard, loss_A_soft, loss_B_hard, loss_B_soft, probs_A, probs_B)

def cal_total_loss(logit_A, logit_B, label_weights, one_hot_labels):
    probs_A1 = tf.nn.softmax(logit_A, -1)
    probs_B1 = tf.nn.softmax(logit_B, -1)
    log_probs_A1 = tf.log(probs_A1)
    log_probs_B1 = tf.log(probs_B1)
    log_probs_A0 = tf.log(1 - probs_A1)
    log_probs_B0 = tf.log(1 - probs_B1)
    probs_A1_stop_gradient = tf.stop_gradient(probs_A1, name="probs_A1_stop_gradient")
    probs_B1_stop_gradient = tf.stop_gradient(probs_B1, name="probs_B1_stop_gradient")
    denominator = tf.reduce_sum(label_weights) + 1e-5

    loss_hard, top_order = cal_hard_loss([log_probs_A1, log_probs_B1], one_hot_labels, label_weights, denominator)

    loss_A_soft = cal_soft_loss(probs_B1_stop_gradient, one_hot_labels, log_probs_A0, log_probs_A1, label_weights, denominator, top_order)
    loss_B_soft = cal_soft_loss(probs_A1_stop_gradient, one_hot_labels, log_probs_B0, log_probs_B1, label_weights, denominator, top_order)
    return loss_hard[0], loss_A_soft, loss_hard[1], loss_B_soft, probs_A1, probs_B1


def cal_soft_loss(tag_prob, one_hot_labels, src_log_probs_0, src_log_probs_1, label_weights, denominator, top_order):
    gamma = FLAGS.gamma
    loss = one_hot_labels * tf.pow(1 - tag_prob, gamma) * src_log_probs_1 + (1 - one_hot_labels) * tf.pow(tag_prob, gamma) * src_log_probs_0
    loss = top_order*label_weights * (-tf.reduce_sum(loss, axis=[-1]))
    numerator = tf.reduce_sum(loss)
    loss = numerator / denominator
    return loss

def cal_hard_loss(src_log_probs_list, one_hot_labels, label_weights, denominator):
    loss_list = []
    for src_log_probs in src_log_probs_list:
        loss_tmp = one_hot_labels * src_log_probs
        loss_tmp = label_weights * (-tf.reduce_sum(loss_tmp, axis=[-1]))
        loss_list.append(loss_tmp)

    beta = FLAGS.beta
    shape_number = tf.shape(loss_list[0])[0]
    top_n = tf.cast(tf.cast(shape_number, tf.float32) * (1 - beta), tf.int32)
    top_order = None
    flag = False
    for loss in loss_list:
        order = tf.nn.top_k(-tf.nn.top_k(-loss,shape_number)[1],shape_number)[1]
        top_order = tf.logical_and(order > top_n, top_order) if flag else (order > top_n)
        flag = True

    top_order = tf.cast(tf.logical_not(top_order), tf.float32)
    loss_hard = []
    for loss in loss_list:
        numerator = tf.reduce_sum(top_order*loss)
        loss_hard.append(numerator / denominator)
    return loss_hard, top_order


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "info":
            tf.io.FixedLenFeature([1], tf.int64),  #[user]
            "input_ids":
            tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
            tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32)
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_files)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        #     # `cycle_length` is the number of parallel files that get read.
        #     cycle_length = min(num_cpu_threads, len(input_files))
        #
        #     # `sloppy` mode means that the interleaving is not exact. This adds
        #     # even more randomness to the training pipeline.
        #     d = d.apply(
        #        tf.contrib.data.parallel_interleave(
        #            tf.data.TFRecordDataset,
        #            sloppy=is_training,
        #            cycle_length=cycle_length))
        #     d = d.shuffle(buffer_size=100)
        # else:
        #     d = tf.data.TFRecordDataset(input_files)

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size))

        # d = d.map(
        #     lambda record: _decode_record(record, name_to_features),
        #     num_parallel_calls=num_cpu_threads)
        # d = d.batch(batch_size=batch_size)
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def main(_):
    session_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session_config.gpu_options.allow_growth = True
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.logging.set_verbosity(tf.logging.INFO)

    FLAGS.checkpointDir = FLAGS.checkpointDir + FLAGS.signature
    print('checkpointDir:', FLAGS.checkpointDir)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.checkpointDir)

    train_input_files = []
    for input_pattern in FLAGS.train_input_file.split(","):
        train_input_files.extend(tf.gfile.Glob(input_pattern))

    test_input_files = []
    if FLAGS.test_input_file is None:
        test_input_files = train_input_files
    else:
        for input_pattern in FLAGS.test_input_file.split(","):
            test_input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** train Input Files ***")
    for input_file in train_input_files:
        tf.logging.info("  %s" % input_file)

    tf.logging.info("*** test Input Files ***")
    for input_file in train_input_files:
        tf.logging.info("  %s" % input_file)

    tpu_cluster_resolver = None

    #is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    if FLAGS.num_gpu_cores >= 2:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        run_config = tf.estimator.RunConfig(
            session_config=session_config,
            model_dir=FLAGS.checkpointDir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            train_distribute=mirrored_strategy)
    else:
        run_config = tf.estimator.RunConfig(
            session_config=session_config,
            model_dir=FLAGS.checkpointDir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    
    if FLAGS.vocab_filename is not None:
        with open(FLAGS.vocab_filename, 'rb') as input_file:
            vocab = pickle.load(input_file)
    item_size = len(vocab.counter)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        item_size=item_size)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={
            "batch_size": FLAGS.batch_size
        })

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        train_input_fn = input_fn_builder(
            input_files=train_input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True)
        estimator.train(
            input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        eval_input_fn = input_fn_builder(
            input_files=test_input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False)

        #tf.logging.info('special eval ops:', special_eval_ops)
        result = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=None,
            hooks=[EvalHooks()])

        output_eval_file = os.path.join(FLAGS.checkpointDir,
                                        "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            tf.logging.info(bert_config.to_json_string())
            writer.write(bert_config.to_json_string()+'\n')
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("checkpointDir")
    flags.mark_flag_as_required("user_history_filename")
    tf.app.run()
