# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import os
import argparse
import time
from model.dataset import ValDataProvider
from model.cgrn import CGRN
from model.utils import compile_frames_to_gif

parser = argparse.ArgumentParser(description='Inference for unseen data')
parser.add_argument('--model_dir', dest='model_dir', required=True,
                    help='directory that saves the model checkpoints')
parser.add_argument('--experiment_dir', dest='experiment_dir', required=False,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--val_obj', dest='val_obj', type=str, required=False, help='the source images for inference')
parser.add_argument('--test_obj', dest='test_obj', type=str, required=False, help='the source images for inference')
parser.add_argument('--embedding_ids', default='embedding_ids', type=str, help='embeddings involved')
parser.add_argument('--save_dir', default='save_dir', type=str, help='path to save inferred images')
parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
                    help='use conditional instance normalization in your model')
parser.add_argument('--interpolate', dest='interpolate', type=int, default=0,
                    help='interpolate between different embedding vectors')
parser.add_argument('--steps', dest='steps', type=int, default=10, help='interpolation steps in between vectors')
parser.add_argument('--output_gif', dest='output_gif', type=str, default=None, help='output name transition gif')
parser.add_argument('--uroboros', dest='uroboros', type=int, default=0,
                    help='Shōnen yo, you have stepped into uncharted territory')
parser.add_argument('--fontclass_num', dest='fontclass_num', type=int, default=1,
                    help='fontclass_num')
parser.add_argument('--charclass_num', dest='charclass_num', type=int, default=62,
                    help='charclass_num')
parser.add_argument('--use_stn', dest='use_stn', type=int, default=1,
                    help='whether to use STN')
parser.add_argument('--use_bn', dest='use_bn', type=int, default=0,
                    help='whether to use BN')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=1,
                    help='gpu_id')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = CGRN(batch_size=args.batch_size, fontclass_num = args.fontclass_num, charclass_num = args.charclass_num, use_stn = args.use_stn, use_bn = args.use_bn)
        model.register_session(sess)
        model.build_model(is_training=False, inst_norm=args.inst_norm)
        test_provider = ValDataProvider(args.test_obj)
        FileName = args.model_dir + '/checkpoint'
        while(True):
            if (os.path.exists(FileName)):
                break
        print('test accuracy')
        model.test_model(model_dir=args.model_dir, val_provider = test_provider)
        print('\n')
        ctime = int(os.stat(FileName).st_ctime)
        while(True):
            if int(os.stat(FileName).st_ctime) != ctime:
                ctime = int(os.stat(FileName).st_ctime)
                print('test accuracy')
                model.test_model(model_dir=args.model_dir, val_provider = test_provider)
                print('\n')
if __name__ == '__main__':
    tf.app.run()