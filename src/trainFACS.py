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
from tensorflow.python.ops import data_flow_ops

HOME = os.path.expanduser("~")

class args():
    model_def =  'models.inception_resnet_v1_2pic'
    logs_base_dir = '~/dev/emoTrans/logs'
    models_base_dir = '~/dev/emoTrans'
    seed = 666
    data_dir = HOME + "/datasets/BosphorusDB_extracted/train"
    val_dir = HOME + "/datasets/BosphorusDB_extracted/val"
    random_crop = True
    image_size = 160
    batch_size = 90
    keep_probability = 1.0
    embedding_size = 200
    weight_decay = 0.00001
    learning_rate_decay_epochs = 1.0
    learning_rate = 0.0001
    epoch_size = 1000
    learning_rate_decay_factor = 1.0
    optimizer = 'ADAGRAD'
    moving_average_decay = 0.9999
    gpu_memory_fraction = 1.0
    pretrained_model = None
    max_nrof_epochs = 500
    people_per_batch = None
    images_per_person = None
    validation_batch_num = 3
    learning_rate_schedule_file = None


def main():
    network = importlib.import_module(args.model_def, 'inference')
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    np.random.seed(seed=args.seed)
    train_set = utility.get_dataset(args.data_dir)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_paths_source_placeholder = tf.placeholder(tf.string, name='image_source_paths')

        image_paths_target_placeholder = tf.placeholder(tf.string, name='image_target_paths')

        labels_placeholder = tf.placeholder(tf.float32, shape=(None, args.embedding_size), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.string, tf.float32],
                                              shapes=[(), (), (args.embedding_size)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_source_placeholder,
                                   image_paths_target_placeholder, labels_placeholder])

        nrof_preprocess_threads = 4
        images_pair_and_labels = []
        for _ in range(nrof_preprocess_threads):
            source_filename, target_filename, label = input_queue.dequeue()
            processed_pair = []
            for filename in [source_filename, target_filename]:
                file_contents = tf.read_file(filename)
                image = tf.image.decode_png(file_contents)

                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                #     WE MAY NOT WANT TO FLIP
                # if args.random_flip:
                #     image = tf.image.random_flip_left_right(image)

                # pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                processed_pair.append(tf.image.per_image_standardization(image))
            images_pair_and_labels.append([[processed_pair[0]], [processed_pair[1]], [label]])
        source_batch, target_batch, labels_batch = tf.train.batch_join(
            images_pair_and_labels, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3),
                    (args.image_size, args.image_size, 3), (args.embedding_size)], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        prelogits, _ = network.inference(source_batch, target_batch, args.keep_probability,
            phase_train = phase_train_placeholder, bottleneck_layer_size = args.embedding_size,
            weight_decay = args.weight_decay)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # sourceIMG, targetIMG = tf.unstack(tf.reshape(embeddings, [-1,2, args.embedding_size]), 2, 1)
        # FACS of source and target
        # sourceLabels, targetLabels = tf.unstack(tf.reshape(labels_batch, [-1,2, args.embedding_size]), 2, 1)
        trans_loss = utility.trans_loss(embeddings, labels_batch)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size,
                                                   args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([trans_loss] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = utility.train(total_loss, global_step, args.optimizer,
                                 learning_rate, args.moving_average_decay, tf.global_variables())

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, train_set, epoch, image_paths_source_placeholder,
                      image_paths_target_placeholder,
                      labels_placeholder, labels_batch,batch_size_placeholder,
                      learning_rate_placeholder, phase_train_placeholder, enqueue_op,
                      input_queue, global_step,embeddings, total_loss, train_op,
                      summary_op, summary_writer, args.learning_rate_schedule_file,
                      args.embedding_size, trans_loss)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver,
                     summary_writer, model_dir, subdir, step)

                # Evaluate on itself
                if args.val_dir:
                    evaluate(sess, enqueue_op, image_paths_source_placeholder,
                      image_paths_target_placeholder, labels_placeholder,labels_batch,
                             phase_train_placeholder, batch_size_placeholder, embeddings, trans_loss,
                             labels_batch, args.val_dir, args.lfw_batch_size, log_dir,
                             step, summary_writer)
    sess.close()


def train(args, sess, dataset, epoch, image_paths_source_placeholder, image_paths_target_placeholder,
          labels_placeholder,labels_batch, batch_size_placeholder, learning_rate_placeholder,
          phase_train_placeholder, enqueue_op, input_queue, global_step,
          embeddings, loss, train_op, summary_op, summary_writer,
          learning_rate_schedule_file, embedding_size, trans_loss):
    batch_number = 0
    batch_size = args.batch_size
    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = utility.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        image_source_paths, image_target_paths, label_trans_array = \
            utility.sample_pair(args.embedding_size, dataset, args.batch_size, args.people_per_batch);

        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()

        sess.run(enqueue_op, {image_paths_source_placeholder: image_source_paths,
                      image_paths_target_placeholder: image_target_paths, labels_placeholder: label_trans_array})
        start_time = time.time()
        feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr,
                     phase_train_placeholder: True}
        err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch],
                                          feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
              (epoch, batch_number + 1, args.epoch_size, duration, err))
        batch_number += 1

        # Add validation loss and accuracy to summary
        summary = tf.Summary()
        # pylint: disable=maybe-no-member
        # summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
    return step


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)

  # evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder,
  #                            phase_train_placeholder, batch_size_placeholder, embeddings,
  #                            labels_batch, args.val_dir, args.lfw_batch_size, log_dir,
  #                            step, summary_writer)
def evaluate(sess, enqueue_op, image_paths_source_placeholder, image_paths_target_placeholder,
         labels_placeholder, labels_batch,phase_train_placeholder, batch_size_placeholder, embeddings,
         trans_loss, labels, image_paths, batch_size, nrof_folds, log_dir, step, summary_writer):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on validation images')

    batch_number = 0
    batch_size = args.batch_size
    dataset = utility.get_dataset(image_paths)
    sumloss = 0
    while batch_number < args.validation_batch_num:
        # Sample people randomly from the dataset
        image_source_paths, image_target_paths, label_trans_array = \
            utility.sample_pair(args.embedding_size, dataset, batch_size, args.people_per_batch);

        sess.run(enqueue_op, {image_paths_source_placeholder: image_source_paths,
                      image_paths_target_placeholder: image_target_paths, labels_placeholder: label_trans_array})
        start_time = time.time()
        feed_dict = {batch_size_placeholder: batch_size, phase_train_placeholder: False}
        err, emb, lab = sess.run([trans_loss, embeddings, labels_batch],
                                          feed_dict=feed_dict)
        sumloss += err
        duration = time.time() - start_time
        batch_number += 1
    avg_loss = sumloss / batch_number


    print('Validation loss: %1.3f+-%1.3f' % avg_loss)
    val_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='val/loss', simple_value=avg_loss)
    summary_writer.add_summary(summary, step)

    with open(os.path.join(log_dir, 'val_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t\n' % (step, avg_loss))



if __name__ == '__main__':
    main()