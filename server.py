# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work
# --------------------------------------------------------
# Train a dense captioning model
# Code adapted from faster R-CNN project
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os.path import join as pjoin
import sys
import six
import glob
import argparse
import json
import numpy as np
import tensorflow as tf
from lib.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from lib.datasets.factory import get_imdb
import lib.datasets.imdb
from lib.dense_cap.train import get_training_roidb, train_net
from lib.dense_cap.test import im_detect, sentence
from lib.nets.vgg16 import vgg16
from lib.nets.resnet_v1 import resnetv1
import pprint
from lib.fast_rcnn.nms_wrapper import nms
import runway


@runway.setup
def setup():
    global net, vocab
    ckpt_dir = 'output/ckpt/'
    vocabulary = 'output/ckpt/vocabulary.txt'

    # load network
    net = resnetv1(num_layers=50) # vgg16() resnetv1(num_layers=50, 101, 152)
    net.create_architecture("TEST", num_classes=1, tag='pre')
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    # load vocab
    vocab = ['<PAD>', '<SOS>', '<EOS>']
    with open(vocabulary, 'r') as f:
        for line in f:
            vocab.append(line.strip())

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    saver = tf.train.Saver()
    sess = tf.Session(config=tfconfig)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Restored from {}'.format(ckpt.model_checkpoint_path))

    return sess

def convert_rect(rect, width, height):
    x = rect[0] / width
    y = rect[1] / height
    w = rect[2] / width
    h = rect[3] / height
    return x, y, w, h
  

caption_inputs = {
    'image': runway.image,
    'max_detections': runway.number(default=10, min=1, max=50, step=1)
}

caption_outputs = {
    'bboxes': runway.array(runway.image_bounding_box),
    'classes': runway.array(runway.text),
    'scores': runway.array(runway.number)
}

@runway.command('caption', inputs=caption_inputs, outputs=caption_outputs)
def caption(sess, inp):
    img = np.array(inp['image'])
    width = img.shape[1]
    height = img.shape[0]
    scores, boxes, captions = im_detect(sess, net, img, None, use_box_at=-1)
    pos_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(pos_dets, cfg.TEST.NMS)
    pos_dets = pos_dets[keep, :]
    pos_scores = scores[keep]
    pos_captions = [sentence(vocab, captions[idx]) for idx in keep]
    pos_boxes = boxes[keep, :]
    bboxes = []
    classes = []
    scores = []
    for i in range(min(inp['max_detections'], len(pos_captions))):
        bboxes.append(convert_rect(pos_boxes[i], width, height))
        classes.append(pos_captions[i])
        scores.append(float(pos_scores[i]))
    return dict(bboxes=bboxes, classes=classes, scores=scores)


if __name__ == '__main__':
    runway.run()
