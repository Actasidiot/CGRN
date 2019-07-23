# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import argparse
import os
from model.cgrn import CGRN
parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--experiment_dir', dest='experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--experiment_id', dest='experiment_id', type=int, default=0,
                    help='sequence id for the experiments you prepare to run')
parser.add_argument('--image_size', dest='image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', dest='L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lcont_penalty', dest='Lcont_penalty', type=int, default=100, help='weight for content loss')
parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--schedule', dest='schedule', type=int, default=10, help='number of epochs to half learning rate')
parser.add_argument('--resume', dest='resume', type=int, default=1, help='resume from previous training')
parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
                    help='use conditional instance normalization in your model')
parser.add_argument('--sample_steps', dest='sample_steps', type=int, default=10,
                    help='number of batches in between two samples are drawn from validation set')
parser.add_argument('--checkpoint_steps', dest='checkpoint_steps', type=int, default=584,
                    help='number of batches in between two checkpoints')
parser.add_argument('--fontclass_num', dest='fontclass_num', type=int, default=4,
                    help='fontclass_num')
parser.add_argument('--charclass_num', dest='charclass_num', type=int, default=1438,
                    help='charclass_num')
parser.add_argument('--use_bn', dest='use_bn', type=int, default=0,
                    help='whether to use BN')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=1,
                    help='gpu_id')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
def main(_):
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = CGRN(args.experiment_dir, batch_size=args.batch_size, experiment_id=args.experiment_id, fontclass_num = args.fontclass_num, charclass_num = args.charclass_num,
                     input_width=args.image_size, output_width=args.image_size,
                     embedding_dim=args.embedding_dim, L1_penalty=args.L1_penalty, Lcont_penalty=args.Lcont_penalty,use_bn = args.use_bn)
        model.register_session(sess)
        model.build_model(is_training=True, inst_norm=args.inst_norm)
        model.train(lr=args.lr, epoch=args.epoch, resume=args.resume,
                    schedule=args.schedule,
                    sample_steps=args.sample_steps, checkpoint_steps=args.checkpoint_steps)


if __name__ == '__main__':
    tf.app.run()
