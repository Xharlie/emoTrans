
import os
import numpy as np
import random
import re
import tensorflow as tf

HOME = os.path.expanduser("~")

class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_dataset(paths):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir, img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))

    return dataset


def sample_pair(dataset, batch_size, people_per_batch):
    nrof_classes = len(dataset)
    if people_per_batch is None or nrof_classes < people_per_batch:
        people_per_batch = min(nrof_classes, batch_size)

    # Sample classes from the dataset
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    avg_pair_per_people = batch_size / people_per_batch
    i = 0
    image_pair_paths = []
    label_calculated_result = []
    codes_array = load_facscodes()
    # Sample images from these classes until we have enough
    while len(image_pair_paths) < batch_size:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        pair_per_people = min(batch_size - len(image_pair_paths), avg_pair_per_people)
        for j in range(pair_per_people):
            indice1 = random.randint(0, nrof_images_in_class - 1)
            indice2 = random.randint(0, nrof_images_in_class - 1)
            image_paths_for_class = [dataset[class_index].image_paths[indice1],
                                     dataset[class_index].image_paths[indice2]]
            label_calculated_result += calculate_labels(codes_array,
                dataset[class_index].image_paths[indice1], dataset[class_index].image_paths[indice2])
            image_pair_paths += image_paths_for_class
        i += 1
    return image_pair_paths, label_calculated_result

def load_facscodes():
    codes_array = {}
    with open(HOME + '/dev/emoTrans/facscodes.lst') as fp:
        for line in fp:
            pair = line.split("->")
            codes_array[pair[0].strip()] = pair[1].strip()
    return codes_array

def calculate_labels(codes_array, image_path1, image_path2):
    calculate_codes_1 = embedding_translate(image_path1)
    calculate_codes_2 = embedding_translate(image_path2)
    return calculate_codes_2 - calculate_codes_1

def embedding_translate(image_path):
    codes = np.zeros(shape = (200), dtype='float32')
    dimensionList = image_path.split("+")
    for code_expression in dimensionList:
        code = int(re.findall('\d+', code_expression)[0])
        alphabet = code_expression.split(str(code))
        pos = 0
        strength = 0
        if len(alphabet) == 1:
            strength = alphabet[0]
        elif alphabet[0] == 'L':
            pos = 1
            strength = alphabet[1]
        elif alphabet[0] == 'R':
            pos = 2
            strength = alphabet[1]
        else:
            strength = alphabet[1]
        strength_float = {
            'A': 0.2,
            'B': 0.4,
            'C': 0.6,
            'D': 0.8,
            'E': 1.0
        }.get(strength)
        if pos == 0:
            codes[code - 1] = strength_float
            codes[code + 100] = strength_float
        elif pos == 1:
            codes[code] = strength_float
        elif pos == 2:
            codes[code + 100] = strength_float
        else:
            print "Wrong!"
    print codes


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate

def trans_loss(embeddings, labels_batch):
    #Calculate the trans loss
    with tf.variable_scope('trans_loss'):
        dist = tf.reduce_sum(tf.square(tf.subtract(embeddings, labels_batch)))
    return dist


def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars,
          log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def _add_loss_summaries(total_loss):
    """Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

if __name__ == '__main__':
    load_facscodes()
