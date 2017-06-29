from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import itertools
import argparse
import utility

HOME = os.path.expanduser("~")
model_def = 'models.inception_resnet_v1'
logs_base_dir='~/dev/emoTrans/'
seed = 666
data_dir = HOME + "/datasets/BosphorusDB"

def main():
    network = importlib.import_module(model_def, 'inference')
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(logs_base_dir), subdir)
    np.random.seed(seed=seed)
    train_set = utility.get_dataset(data_dir)

    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name='image_paths')

        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(3,), (3,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

        nrof_preprocess_threads = 4
        images_and_labels = []

if __name__ == '__main__':
    main()