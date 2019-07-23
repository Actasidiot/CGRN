# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os
import glob

import imageio
import scipy.misc as misc
import numpy as np
from cStringIO import StringIO


def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq


def bytes_to_file(bytes_img):
    return StringIO(bytes_img)


def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    #normalized = img - 127.5
    return normalized


def read_split_image(img, fontclass_num):
    #(64, 320, 3)
    mat = misc.imread(img).astype(np.float)
    side = int(mat.shape[1] / (fontclass_num + 1))
    assert side * (fontclass_num + 1) == mat.shape[1]
    img_list = []
    for idx in range(0, fontclass_num + 1):
        #mat[:, idx*side:(idx+1)*side]
        img_list.append(mat[:, idx*side:(idx+1)*side])  # target
        #img_B = mat[:, side:]  # source
    #print(len(img_list))
    #print(img_list[0])
    #raw_input()
    return img_list


def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    w, h, _ = img.shape
    enlarged = misc.imresize(img, [nw, nh])
    return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]


def scale_back(images):
    return 255.0 * (images + 1.) / 2.
    #return images + 127.5

def scale_back_for_fake(images):
    return 255.0 * (images + 1.) / 2.
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
   
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def save_concat_images(imgs, img_path):
    concated = np.concatenate(imgs, axis=1)
    misc.imsave(img_path, concated)


def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(frames)
    images = [misc.imresize(imageio.imread(f), interp='nearest', size=0.33) for f in frames]
    imageio.mimsave(gif_file, images, duration=0.1)
    return gif_file
