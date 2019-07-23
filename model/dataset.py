# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import cPickle as pickle
import numpy as np
import random
import os
from .utils import pad_seq, bytes_to_file, \
    read_split_image, shift_and_resize_image, normalize_image


class PickledImageProvider(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                    if len(examples) % 1000 == 0:
                        print("processed %d examples" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            print("unpickled total %d examples" % len(examples))
            return examples


def get_batch_iter(examples, batch_size, fontclass_num, augment):
    # the transpose ops requires deterministic
    # batch size, thus comes the padding
    fontclass_num = 4
    padded = pad_seq(examples, batch_size)
    def process(img,random_idx):
        img = bytes_to_file(img)
        try:
            img_list = read_split_image(img, fontclass_num)
            img_seq_np = np.array(normalize_image(img_list[0]))
            glyph_img = normalize_image(img_list[random_idx + 1])
            img_seq_np = np.concatenate([img_seq_np, glyph_img], axis=2)
            #img_A, img_B = read_split_image(img)
            if augment:
                # augment the image by:
                # 1) enlarge the image
                # 2) random crop the image back to its original size
                # NOTE: image A and B needs to be in sync as how much
                # to be shifted
                w, h, _ = img_A.shape
                multiplier = random.uniform(1.00, 1.20)
                # add an eps to prevent cropping issue
                nw = int(multiplier * w) + 1
                nh = int(multiplier * h) + 1
                shift_x = int(np.ceil(np.random.uniform(0.01, nw - w)))
                shift_y = int(np.ceil(np.random.uniform(0.01, nh - h)))
                img_A = shift_and_resize_image(img_A, shift_x, shift_y, nw, nh)
                img_B = shift_and_resize_image(img_B, shift_x, shift_y, nw, nh)
            return img_seq_np
        finally:
            img.close()

    def label_process(labelstr,random_idx):

        font_label = int(labelstr[random_idx])
        return font_label

    def batch_iter():
        for i in range(0, len(padded), batch_size):
            batch = padded[i: i + batch_size]
            font_labels = []
            char_labels = []
            img_names = []
            processed = []
            for e in batch:
                #shuffled_order = range(fontclass_num)
                random_idx = random.randint(0,fontclass_num-1)
                #random.shuffle(shuffled_order)
                font_labels.append(label_process(e[0],random_idx))
                char_labels.append(e[1])
                img_names.append(e[2])
                processed.append(process(e[3],random_idx))
            #font_labels = [label_process(e[0]) for e in batch]
            #char_labels = [e[1] for e in batch]
            #img_names = [e[2] for e in batch]
            #processed = [process(e[3]) for e in batch]
            # stack into tensor
            yield np.array(font_labels), char_labels, img_names, np.array(processed).astype(np.float32)

    return batch_iter()


class TrainDataProvider(object):
    def __init__(self, data_dir, train_name="train.obj", val_name="val.obj"):
        self.data_dir = data_dir
        self.train_path = os.path.join(self.data_dir, train_name)
        self.train = PickledImageProvider(self.train_path)
        print("train examples -> %d" % (len(self.train.examples)))

    def get_train_iter(self, batch_size, fontclass_num, shuffle=True):
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.shuffle(training_examples)
        return get_batch_iter(training_examples, batch_size, fontclass_num, augment=False)

    def get_val_iter(self, batch_size, fontclass_num, shuffle=False):
        """
        Validation iterator runs forever
        """
        val_examples = self.val.examples[:]
        if shuffle:
            np.random.shuffle(val_examples)

        return get_batch_iter(val_examples, batch_size, fontclass_num, augment=False)

    def compute_total_batch_num(self, batch_size):
        """Total padded batch num"""
        return int(np.ceil(len(self.train.examples) / float(batch_size)))

    def get_all_labels(self):
        """Get all training labels"""
        return list({e[0] for e in self.train.examples})

    def get_train_val_path(self):
        return self.train_path, self.val_path


class InjectDataProvider(object):
    def __init__(self, obj_path):
        self.data = PickledImageProvider(obj_path)
        print("examples -> %d" % len(self.data.examples))

    def get_single_embedding_iter(self, batch_size, fontclass_num, embedding_id):
        examples = self.data.examples[:]
        batch_iter = get_batch_iter(examples, batch_size, fontclass_num, augment=False)
        for font_labels, char_labels, img_names, images in batch_iter:
            # inject specific embedding style here
            labels = [embedding_id] * batch_size
            yield font_labels, char_labels, img_names, images

    def get_random_embedding_iter(self, batch_size, fontclass_num, embedding_ids):
        examples = self.data.examples[:]
        batch_iter = get_batch_iter(examples, batch_size, fontclass_num, augment=False)
        for font_labels, char_labels, img_names, images in batch_iter:
            # inject specific embedding style here
            labels = [random.choice(embedding_ids) for i in range(batch_size)]
            yield font_labels, char_labels, img_names, images


class NeverEndingLoopingProvider(InjectDataProvider):
    def __init__(self, obj_path):
        super(NeverEndingLoopingProvider, self).__init__(obj_path)

    def get_random_embedding_iter(self, batch_size, embedding_ids):
        while True:
            # np.random.shuffle(self.data.examples)
            rand_iter = super(NeverEndingLoopingProvider, self) \
                .get_random_embedding_iter(batch_size, embedding_ids)
            for font_labels, char_labels, img_names, images in rand_iter:
                yield font_labels, char_labels, img_names, images
class ValDataProvider(object):
    def __init__(self, obj_path):
        self.val = PickledImageProvider(obj_path)
        print("examples -> %d" % len(self.val.examples))
    def get_val_iter(self, batch_size, fontclass_num, shuffle=False):
        """
        Validation iterator runs forever
        """
        val_examples = self.val.examples[:]
        if shuffle:
            np.random.shuffle(val_examples)
        return get_batch_iter(val_examples, batch_size, fontclass_num, augment=False)
