# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import time
from PIL import Image
from collections import namedtuple
from .ops import conv2d, deconv2d, lrelu, fc, batch_norm, init_embedding, conditional_instance_norm
from .dataset import TrainDataProvider, InjectDataProvider, NeverEndingLoopingProvider, ValDataProvider
from .utils import scale_back, merge, save_concat_images, scale_back_for_fake

# Auxiliary wrapper classes
# Used to save handles(important nodes in computation graph) for later evaluation
LossHandle = namedtuple("LossHandle", ["g_loss",  "cont_loss", "l1_loss",  "d_loss",  "cheat_loss"])
InputHandle = namedtuple("InputHandle", ["real_data", "embedding_ids", "char_classes"])
EvalHandle = namedtuple("EvalHandle", ["encoder", "generator", "target", "source", "embedding","cont_fc"])
SummaryHandle = namedtuple("SummaryHandle", ["g_merged"])


class CGRN(object):
    def __init__(self, experiment_dir=None, experiment_id=0, batch_size=16, fontclass_num = 4, input_width=64, output_width=64,
                 generator_dim=64, discriminator_dim=64, L1_penalty=100, Lcont_penalty=100,
                 embedding_dim=128, charclass_num=62, input_filters=3, output_filters=3, use_stn = 1, use_bn = 0):
        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.batch_size = batch_size
        self.fontclass_num = fontclass_num
        self.input_width = input_width
        self.output_width = output_width
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.L1_penalty = L1_penalty
        self.Lcont_penalty = Lcont_penalty
        self.embedding_dim = embedding_dim
        self.charclass_num = charclass_num
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.use_bn = use_bn
        # init all the directories
        self.sess = None
        # experiment_dir is needed for training
        if experiment_dir:
            self.data_dir = os.path.join(self.experiment_dir, "data")
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
            self.sample_dir = os.path.join(self.experiment_dir, "sample")
            self.log_dir = os.path.join(self.experiment_dir, "logs")

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print("create checkpoint directory")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                print("create log directory")
            if not os.path.exists(self.sample_dir):
                os.makedirs(self.sample_dir)
                print("create sample directory")

    def encoder(self, images, is_training, reuse=False):
        # feature extracor based on VGG16
        # conv1_1
        encode_layers = dict()
        with tf.variable_scope('encoder/conv1_1'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            kernel = tf.get_variable('weights', [3, 3, 3, 64], initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            if self.use_bn:
                out = batch_norm(out, is_training, scope="bn")
            conv1_1 = tf.nn.relu(out)

        # conv1_2
        with tf.variable_scope('encoder/conv1_2'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            kernel = tf.get_variable('weights', [3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            if self.use_bn:
                out = batch_norm(out, is_training, scope="bn")
            conv1_2 = tf.nn.relu(out)

        # pool1
        pool1 = tf.nn.max_pool(conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')
        
        #pool1 = conv1_2
        encode_layers["p1"] = pool1
        # conv2_1
        with tf.variable_scope('encoder/conv2_1'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            kernel = tf.get_variable('weights', [3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            if self.use_bn:
                out = batch_norm(out, is_training, scope="bn")
            conv2_1 = tf.nn.relu(out)
        # conv2_2
        with tf.variable_scope('encoder/conv2_2'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            kernel = tf.get_variable('weights', [3, 3, 128, 128], initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            if self.use_bn:
                out = batch_norm(out, is_training, scope="bn")
            conv2_2 = tf.nn.relu(out)

        # pool2
        pool2 = tf.nn.max_pool(conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')
        
        #pool2 = conv2_2
        encode_layers["p2"] = pool2
        # conv3_1
        with tf.variable_scope('encoder/conv3_1'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            kernel = tf.get_variable('weights', [3, 3, 128, 256], initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            if self.use_bn:
                out = batch_norm(out, is_training, scope="bn")
            conv3_1 = tf.nn.relu(out)

        # conv3_2
        with tf.variable_scope('encoder/conv3_2'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            kernel = tf.get_variable('weights', [3, 3, 256, 256], initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            if self.use_bn:
                out = batch_norm(out, is_training, scope="bn")
            conv3_2 = tf.nn.relu(out)

        # conv3_3
        with tf.variable_scope('encoder/conv3_3'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            kernel = tf.get_variable('weights', [3, 3, 256, 256], initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            if self.use_bn:
                out = batch_norm(out, is_training, scope="bn")
            conv3_3 = tf.nn.relu(out)
            #parameters += [kernel, biases]

        # pool3
        pool3 = tf.nn.max_pool(conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')
        
        #pool3 = conv3_3
        encode_layers["p3"] = pool3
        # conv4_1
        with tf.variable_scope('encoder/conv4_1'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            kernel = tf.get_variable('weights', [3, 3, 256, 512], initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            if self.use_bn:
                out = batch_norm(out, is_training, scope="bn")
            conv4_1 = tf.nn.relu(out)

        # conv4_2
        with tf.variable_scope('encoder/conv4_2'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            kernel = tf.get_variable('weights', [3, 3, 512, 512], initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            if self.use_bn:
                out = batch_norm(out, is_training, scope="bn")
            conv4_2 = tf.nn.relu(out)

        # conv4_3
        with tf.variable_scope('encoder/conv4_3'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            kernel = tf.get_variable('weights', [3, 3, 512, 512], initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            if self.use_bn:
                out = batch_norm(out, is_training, scope="bn")
            conv4_3 = tf.nn.relu(out)

        # pool4
        pool4 = tf.nn.max_pool(conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')
        
        #pool4 = conv4_3
        encode_layers["p4"] = pool4
        # conv5_1
        with tf.variable_scope('encoder/conv5_1'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            kernel = tf.get_variable('weights', [3, 3, 512, 512], initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            if self.use_bn:
                out = batch_norm(out, is_training, scope="bn")
            conv5_1 = tf.nn.relu(out)

        # conv5_2
        with tf.variable_scope('encoder/conv5_2'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            kernel = tf.get_variable('weights', [3, 3, 512, 512], initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            if self.use_bn:
                out = batch_norm(out, is_training, scope="bn")
            conv5_2 = tf.nn.relu(out)

        # conv5_3
        with tf.variable_scope('encoder/conv5_3'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            kernel = tf.get_variable('weights', [3, 3, 512, 512], initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            if self.use_bn:
                out = batch_norm(out, is_training, scope="bn")
            conv5_3 = tf.nn.relu(out)

        # pool5
        pool5 = tf.nn.max_pool(conv5_3,
                               ksize=[1, 4, 4, 1],
                               strides=[1, 4, 4, 1],
                               padding='SAME',
                               name='pool5')
        return pool5,  encode_layers

    def decoder(self, encoded, encoding_layers, ids, inst_norm, is_training, reuse=False):
        with tf.variable_scope("decoder"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            s = self.output_width
            s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
                s / 64), int(s / 128)

            def decode_layer(x, output_width, output_filters, layer, enc_layer, dropout=False, do_concat = True):
                dec = deconv2d(tf.nn.relu(x), [self.batch_size, output_width,
                                               output_width, output_filters], scope="g_d%d_deconv" % layer)
                if layer != 8:
                    # IMPORTANT: normalization for last layer
                    # Very important, otherwise GAN is unstable
                    # Trying conditional instance normalization to
                    # overcome the fact that batch normalization offers
                    # different train/test statistics
                    if inst_norm:
                        dec = conditional_instance_norm(dec, ids, self.embedding_num, scope="g_d%d_inst_norm" % layer)
                    else:
                        dec = batch_norm(dec, is_training, scope="g_d%d_bn" % layer)
                if dropout:
                    dec = tf.nn.dropout(dec, 0.5)
                if do_concat:
                    dec = tf.concat([dec, enc_layer], 3)
                return dec

            d3 = decode_layer(encoded, s32, self.generator_dim * 8, layer=3, dropout=is_training, enc_layer=None, do_concat=False)
            d4 = decode_layer(d3, s16, self.generator_dim * 8, layer=4,dropout=is_training,enc_layer=encoding_layers["p4"])
            d5 = decode_layer(d4, s8, self.generator_dim * 4, layer=5, dropout=is_training,enc_layer=encoding_layers["p3"])
            d6 = decode_layer(d5, s4, self.generator_dim * 2, layer=6,enc_layer=encoding_layers["p2"])
            d7 = decode_layer(d6, s2, self.generator_dim, layer=7, enc_layer=encoding_layers["p1"])
            d8 = decode_layer(d7, s, self.output_filters, layer=8, enc_layer=None, do_concat=False)
            output = tf.nn.tanh(d8)  # scale to (-1, 1)
            return output

    def generator(self, encoded_feat, encoding_layers, embeddings, embedding_ids, inst_norm, is_training, reuse=False):
        local_embeddings = tf.nn.embedding_lookup(embeddings, ids=embedding_ids)
        local_embeddings = tf.reshape(local_embeddings, [self.batch_size, 1, 1, self.embedding_dim])
        embedded = tf.concat([encoded_feat, local_embeddings], 3)
        output = self.decoder(embedded, encoding_layers, embedding_ids, inst_norm, is_training=is_training, reuse=reuse)
        return output, embedded

    def discriminator(self, image, is_training, reuse=False):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            h0 = lrelu(conv2d(image, self.discriminator_dim, scope="d_h0_conv"))
            h1 = lrelu(batch_norm(conv2d(h0, self.discriminator_dim * 2, scope="d_h1_conv"),
                                  is_training, scope="d_bn_1"))
            h2 = lrelu(batch_norm(conv2d(h1, self.discriminator_dim * 4, scope="d_h2_conv"),
                                  is_training, scope="d_bn_2"))
            h3 = lrelu(batch_norm(conv2d(h2, self.discriminator_dim * 8, sh=1, sw=1, scope="d_h3_conv"),
                                  is_training, scope="d_bn_3"))
            # real or fake binary loss
            fc1 = fc(tf.reshape(h3, [self.batch_size, -1]), 1, scope="d_fc1")
            # category loss
            return tf.nn.sigmoid(fc1), fc1

    def build_model(self, is_training=True, inst_norm=False, no_target_source=False):
        real_data = tf.placeholder(tf.float32,
                                   [self.batch_size, self.input_width, self.input_width,
                                    self.input_filters + self.output_filters ],
                                   name='real_A_and_B_images')
        embedding_ids = tf.placeholder(tf.int64, shape=None, name="embedding_ids")
        char_classes = tf.placeholder(tf.int64, shape=None, name="char_classes")

        # source images
        real_A = real_data[:, :, :, 0 : self.input_filters]
        # target images
        real_B = real_data[:, :, :, self.input_filters : self.input_filters + self.output_filters]
        encoded_real_A, encode_layers = self.encoder(real_A, is_training=is_training, reuse=False)
        #embedding = init_embedding(self.fontclass_num, self.embedding_dim)
        embedding = init_embedding(self.fontclass_num, self.embedding_dim, scope="decoder/embedding")
        fake_B, embedded = self.generator(encoded_real_A, encode_layers,  embedding, embedding_ids, is_training=is_training, inst_norm=inst_norm, reuse=False)

        real_AB = tf.concat([real_A, real_B], 3)
        fake_AB = tf.concat([real_A, fake_B], 3)

        pool1_p = tf.nn.max_pool(encode_layers["p1"],
                       ksize=[1, 32, 32, 1],
                       strides=[1, 32, 32, 1],
                       padding='SAME',
                       name='pool1_p')
        pool2_p = tf.nn.max_pool(encode_layers["p2"],
                       ksize=[1, 16, 16, 1],
                       strides=[1, 16, 16, 1],
                       padding='SAME',
                       name='pool1_p')
        pool3_p = tf.nn.max_pool(encode_layers["p3"],
                       ksize=[1, 8, 8, 1],
                       strides=[1, 8, 8, 1],
                       padding='SAME',
                       name='pool3_p')
        pool4_p = tf.nn.max_pool(encode_layers["p4"],
                       ksize=[1, 4, 4, 1],
                       strides=[1, 4, 4, 1],
                       padding='SAME',
                       name='pool4_p')
        encoded_real_A = tf.concat([encoded_real_A,pool1_p],3)
        encoded_real_A = tf.concat([encoded_real_A,pool2_p],3)
        encoded_real_A = tf.concat([encoded_real_A,pool3_p],3)
        encoded_real_A = tf.concat([encoded_real_A,pool4_p],3)
        
        encoded_real_A = tf.reshape(encoded_real_A,[self.batch_size, -1])
        cont_fc = fc(encoded_real_A, self.charclass_num, scope="classifier/g_cont_fc")
        cont_fc_sum = cont_fc

        real_D, real_D_logits = self.discriminator(real_AB, is_training=is_training, reuse=False)
        fake_D, fake_D_logits = self.discriminator(fake_AB, is_training=is_training, reuse=True)
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits,
                                                                         labels=tf.ones_like(real_D)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
                                                                         labels=tf.zeros_like(fake_D)))
        # maximize the chance generator fool the discriminator
        cheat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits, labels=tf.ones_like(fake_D)))

        # encoding contant loss
        # this loss assume that generated imaged and real image
        # should reside in the same space and close to each other
        char_true_classes = tf.reshape(tf.one_hot(indices=char_classes, depth=self.charclass_num),
                                 shape=[self.batch_size, self.charclass_num])


        cont_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cont_fc,labels=char_true_classes)) * self.Lcont_penalty
        # L1 loss between real and generated images
        l1_loss = self.L1_penalty * tf.reduce_mean(tf.abs(fake_B - real_B))
        # total variation loss
        width = self.output_width


        d_loss = d_loss_real + d_loss_fake
        g_loss = l1_loss  + cont_loss +  cheat_loss  
        l1_loss_summary = tf.summary.scalar("l1_loss", l1_loss)
        cont_loss_summary = tf.summary.scalar("cont_loss", cont_loss)
        g_loss_summary = tf.summary.scalar("g_loss", g_loss)


        g_merged_summary = tf.summary.merge([l1_loss_summary,
                                             cont_loss_summary,
                                             g_loss_summary])

        # expose useful nodes in the graph as handles globally
        input_handle = InputHandle(real_data=real_data,
                                   embedding_ids=embedding_ids,
                                   char_classes=char_classes)

        loss_handle = LossHandle(g_loss=g_loss,
                                 cont_loss=cont_loss,
                                 d_loss=d_loss,
                                 cheat_loss=cheat_loss,
                                 l1_loss=l1_loss)

        eval_handle = EvalHandle(encoder=encoded_real_A,
                                 generator=fake_B,
                                 target=real_B,
                                 source=real_A,
                                 cont_fc=cont_fc,
                                 embedding=embedding)

        summary_handle = SummaryHandle(g_merged=g_merged_summary)

        # those operations will be shared, so we need
        # to make them visible globally
        setattr(self, "input_handle", input_handle)
        setattr(self, "loss_handle", loss_handle)
        setattr(self, "eval_handle", eval_handle)
        setattr(self, "summary_handle", summary_handle)

    def register_session(self, sess):
        self.sess = sess

    def retrieve_trainable_vars(self):
        t_vars = tf.trainable_variables()
        dis_vars = [var for var in t_vars if ('discriminator' in var.name)]
        enc_clf_dec_vars = [var for var in t_vars if ('encoder' in var.name) or('classifier' in var.name) or ('decoder' in var.name)]
        enc_clf_vars = [var for var in t_vars if ('encoder' in var.name) or ('classifier' in var.name)]
        return enc_clf_vars, enc_clf_dec_vars, dis_vars

    def retrieve_global_vars(self):
        all_vars = tf.global_variables()
        return all_vars

    def retrieve_handles(self):
        input_handle = getattr(self, "input_handle")
        loss_handle = getattr(self, "loss_handle")
        eval_handle = getattr(self, "eval_handle")
        summary_handle = getattr(self, "summary_handle")

        return input_handle, loss_handle, eval_handle, summary_handle

    def get_model_id_and_dir(self):
        model_id = "experiment_%d_batch_%d" % (self.experiment_id, self.batch_size)
        model_dir = os.path.join(self.checkpoint_dir, model_id)
        return model_id, model_dir

    def checkpoint(self, saver, step):
        model_name = "unet.model"
        model_id, model_dir = self.get_model_id_and_dir()

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        saver.save(self.sess, os.path.join(model_dir, model_name), global_step=step)

    def restore_model(self, saver, model_dir):

        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored model %s" % model_dir)
        else:
            print("fail to restore model %s" % model_dir)

    def load_weights(self, sess):
        weights = np.load('./pretrained_vgg/vgg16_weights.npz')
        keys = sorted(weights.keys())
        all_para = tf.trainable_variables()
        parameters = [var for var in all_para if 'conv' in var.name and "generator" not in var.name and "bn" not in var.name]
        for i, k in enumerate(keys):
            if 'conv' in k:
                sess.run(parameters[i].assign(weights[k]))

    def test_model(self, model_dir, val_provider):
        saver = tf.train.Saver(var_list=self.retrieve_global_vars())
        self.restore_model(saver, model_dir)

        val_batch_iter = val_provider.get_val_iter(self.batch_size, self.fontclass_num, shuffle=False)
        input_handle, loss_handle, eval_handle, _ = self.retrieve_handles()
        dict_imgname_logits = {}
        dict_imgname_logits_sum = {}
        dict_imgname_charlabels = {}
        total_cont_loss = 0.0
        for bid, batch in enumerate(val_batch_iter):
            font_labels, char_labels, img_names, batch_images = batch
            cont_fc, fake_B, real_B, real_A, cont_loss  = self.sess.run([eval_handle.cont_fc,  eval_handle.generator, eval_handle.target, eval_handle.source, loss_handle.cont_loss],
                                feed_dict={
                                input_handle.real_data: batch_images,
                                input_handle.embedding_ids: font_labels,
                                input_handle.char_classes: char_labels})
            cont_fc_np = np.array(cont_fc)
            char_labels_np = np.array(char_labels)
            font_labels = np.array(font_labels)
            if bid == 0:
                outim = Image.new("RGB",(self.fontclass_num * self.output_width, self.batch_size * self.output_width))
                outim_real_target = Image.new("RGB",(self.fontclass_num * self.output_width, self.batch_size * self.output_width))
                fake_images_merge = merge(scale_back(fake_B), [self.batch_size, 1])
                real_images_merge = merge(scale_back(real_B), [self.batch_size, 1])
                real_A_merge = merge(scale_back(real_A), [self.batch_size, 1])
                fake_images_merge_ = Image.fromarray(np.uint8(fake_images_merge))
                real_images_merge_ = Image.fromarray(np.uint8(real_images_merge))
                outim.paste(fake_images_merge_,(0, 0, self.output_width, self.output_width * self.batch_size))
                outim_real_target.paste(real_images_merge_, (0, 0, self.output_width , self.output_width * self.batch_size))
                outim.save(model_dir + '/' + 'sample-fake.jpg')
                outim_real_target.save(model_dir + '/' + 'sample-real.jpg')
                np.savetxt(model_dir + '/' + 'char_label.txt', char_labels_np)
                Image.fromarray(np.uint8(real_A_merge)).save(model_dir + '/' + 'sample-input.jpg')
            
            for img_id, img_name in enumerate(img_names):
                if dict_imgname_logits.get(img_name) is None:
                    dict_imgname_logits[img_name] = cont_fc_np[img_id]
                    dict_imgname_charlabels[img_name] = char_labels_np[img_id]
        correct_prediction_cnt = 0
        for imgname, logits in dict_imgname_logits.items():
            if np.argmax(dict_imgname_logits[imgname]) == dict_imgname_charlabels[imgname]:
                correct_prediction_cnt += 1

        print('testing result:')
        print(float(correct_prediction_cnt) / len(dict_imgname_logits))
        #print('contant loss %.8f' % (total_cont_loss))

    def train(self, lr=0.0001, epoch=100, schedule=10, resume=True, sample_steps=50, checkpoint_steps=152):
        enc_clf_vars, enc_clf_dec_vars, dis_vars = self.retrieve_trainable_vars()
        input_handle, loss_handle, _, summary_handle = self.retrieve_handles()

        if not self.sess:
            raise Exception("no session registered")
        learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.d_loss, var_list=dis_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.g_loss, var_list=enc_clf_dec_vars)
        tf.global_variables_initializer().run()
        real_data = input_handle.real_data
        embedding_ids = input_handle.embedding_ids
        char_classes = input_handle.char_classes

        data_provider = TrainDataProvider(self.data_dir)
        total_batches = data_provider.compute_total_batch_num(self.batch_size)
        
        saver = tf.train.Saver(max_to_keep=1, var_list=self.retrieve_global_vars())
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        if resume:
            _, model_dir = self.get_model_id_and_dir()
            self.restore_model(saver, model_dir)
        else:
            self.load_weights(self.sess)

        current_lr = lr
        counter = 0
        start_time = time.time()

        for ei in range(epoch):
            train_batch_iter = data_provider.get_train_iter(self.batch_size, self.fontclass_num)
            '''
            if  (ei + 1)  % schedule == 0:
                update_lr = current_lr
                # minimum learning rate guarantee
                update_lr = max(update_lr, 0.0002)
                print("decay learning rate from %.8f to %.8f" % (current_lr, update_lr))
                current_lr = update_lr
            '''
            for bid, batch in enumerate(train_batch_iter):
                counter += 1
                font_labels, char_labels, img_names, batch_images = batch
                # Optimize D
                _, batch_d_loss = self.sess.run([d_optimizer, loss_handle.d_loss],
                                                           feed_dict={
                                                               real_data: batch_images,
                                                               embedding_ids: font_labels,
                                                               char_classes: char_labels,
                                                               learning_rate: current_lr
                                                           })
                '''
                # Optimize G
                _, batch_g_loss = self.sess.run([g_optimizer, loss_handle.g_loss],
                                                feed_dict={
                                                    real_data: batch_images,
                                                    embedding_ids: font_labels,
                                                    char_classes: char_labels,
                                                    learning_rate: current_lr
                                                })
                
                '''
                # Optimize G
                _, batch_g_loss = self.sess.run([g_optimizer, loss_handle.g_loss],
                                                feed_dict={
                                                    real_data: batch_images,
                                                    embedding_ids: font_labels,
                                                    char_classes: char_labels,
                                                    learning_rate: current_lr
                                                })
                _, batch_g_loss, cheat_loss, \
                cont_loss, l1_loss, g_summary = self.sess.run([g_optimizer,
                                                                         loss_handle.g_loss,
                                                                         loss_handle.cheat_loss,
                                                                         loss_handle.cont_loss,
                                                                         loss_handle.l1_loss,
                                                                         summary_handle.g_merged],
                                                                        feed_dict={
                                                                            real_data: batch_images,
                                                                            embedding_ids: font_labels,
                                                                            char_classes: char_labels,
                                                                            learning_rate: current_lr
                                                                        })
                passed = time.time() - start_time
                log_format = "Epoch: [%2d], Stage2, [%4d/%4d] time: %4.4f,  d_loss: %.5f, g_loss: %.5f, " + \
                             "cheat_loss: %.5f, cont_loss: %.5f, l1_loss: %.5f"
                if bid % 120 == 0:
                    print(log_format % (ei, bid, total_batches, passed,  batch_d_loss, batch_g_loss, cheat_loss, cont_loss, l1_loss))
                summary_writer.add_summary(g_summary, counter)

                if counter % checkpoint_steps == 0:
                    print("Checkpoint: save checkpoint step %d" % counter)
                    self.checkpoint(saver, counter)
        # save the last checkpoint
        print("Checkpoint: last checkpoint step %d" % counter)
        self.checkpoint(saver, counter)
