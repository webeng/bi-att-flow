import gzip
import json
from json import encoder
import os

import tensorflow as tf

from basic.evaluator import Evaluation, F1Evaluation
from my.utils import short_floats

import pickle

import boto3
from boto3.session import Session

session = Session(aws_access_key_id='AKIAJJABZMFEQWGHCKFA',
        aws_secret_access_key='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',
        region_name='eu-west-1')

s3 = session.resource('s3', use_ssl=False, verify=False)
bucket = s3.Bucket('jofihu')
#dir_path = os.path.dirname(os.path.realpath(__file__))


class GraphHandler(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.saver = tf.train.Saver(max_to_keep=config.max_to_keep)
        self.writer = None
        self.save_path = os.path.join(config.save_dir, config.model_name)

    def initialize(self, sess):
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        if self.config.load:
            self._load(sess)

        if self.config.mode == 'train':
            #self.writer = tf.train.SummaryWriter(self.config.log_dir, graph=tf.get_default_graph())
            self.writer = tf.summary.FileWriter(self.config.log_dir, graph=tf.get_default_graph())

    def save(self, sess, global_step=None):
        saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        saver.save(sess, self.save_path, global_step=global_step)
        # Store to S3
        save_src_path = os.getcwd() + '/' + "/".join(self.save_path.split('/')[0:-1])
        for file_name in os.listdir(save_src_path):
            src = save_src_path + '/' + file_name
            dst = 'save/' + file_name
            print("src: {} dest: {}".format(src, dst))
            bucket.upload_file(src, dst)
            # key = config.env + '/' + store_path + '/batch_' + str(batch_index) + '.json'
            # bucket.put_object(Body=json.dumps(r.json()), Key=key)

    def _load(self, sess):
        config = self.config
        #vars_ = {var.name.split(":")[0]: var for var in tf.all_variables()}
        vars_ = {var.name.split(":")[0]: var for var in tf.global_variables()}
        if config.load_ema:
            ema = self.model.var_ema
            for var in tf.trainable_variables():
                del vars_[var.name.split(":")[0]]
                vars_[ema.average_name(var)] = var
        saver = tf.train.Saver(vars_, max_to_keep=config.max_to_keep)

        if config.load_path:
            save_path = config.load_path
        elif config.load_step > 0:
            save_path = os.path.join(config.save_dir, "{}-{}".format(config.model_name, config.load_step))
        else:
            save_dir = config.save_dir
            checkpoint = tf.train.get_checkpoint_state(save_dir)
            assert checkpoint is not None, "cannot load checkpoint at {}".format(save_dir)
            save_path = checkpoint.model_checkpoint_path
        print("Loading saved model from {}".format(save_path))
        saver.restore(sess, save_path)

    def add_summary(self, summary, global_step):
        self.writer.add_summary(summary, global_step)

    def add_summaries(self, summaries, global_step):
        for summary in summaries:
            self.add_summary(summary, global_step)

    def dump_eval(self, e, precision=2, path=None):
        assert isinstance(e, Evaluation)
        if self.config.dump_pickle:
            path = path or os.path.join(self.config.eval_dir, "{}-{}.pklz".format(e.data_type, str(e.global_step).zfill(6)))
            with gzip.open(path, 'wb', compresslevel=3) as fh:
                pickle.dump(e.dict, fh)
        else:
            path = path or os.path.join(self.config.eval_dir, "{}-{}.json".format(e.data_type, str(e.global_step).zfill(6)))
            with open(path, 'w') as fh:
                json.dump(short_floats(e.dict, precision), fh)

    def dump_answer(self, e, path=None):
        assert isinstance(e, Evaluation)
        path = path or os.path.join(self.config.answer_dir, "{}-{}.json".format(e.data_type, str(e.global_step).zfill(6)))
        with open(path, 'w') as fh:
            json.dump(e.id2answer_dict, fh)

