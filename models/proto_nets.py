# models/ProtoNet
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class ProtoNet(tf.keras.Model):

    def __init__(self, num_filters, latent_dim):
        super(ProtoNet, self).__init__()
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        num_filter_list = self.num_filters + [latent_dim]
        self.convs = []
        for i, num_filter in enumerate(num_filter_list):
            block_parts = [
                layers.Conv2D(
                    filters=num_filter,
                    kernel_size=3,
                    padding='SAME',
                    activation='linear'),
            ]

            block_parts += [layers.BatchNormalization()]
            block_parts += [layers.Activation('relu')]
            block_parts += [layers.MaxPool2D()]
            block = tf.keras.Sequential(block_parts, name='conv_block_%d' % i)
            self.__setattr__("conv%d" % i, block)
            self.convs.append(block)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inp, training=True):
        out = inp
        for conv in self.convs:
            out = conv(out, training=training)
        out = self.flatten(out)
        return out


def ProtoLoss(x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries):
    """
      calculates the prototype network loss using the latent representation of x
      and the latent representation of the query set
      Args:
        x_latent: latent representation of supports with shape [N*S, D], where D is the latent dimension
        q_latent: latent representation of queries with shape [N*Q, D], where D is the latent dimension
        labels_onehot: one-hot encodings of the labels of the queries with shape [N, Q, N]
        num_classes: number of classes (N) for classification
        num_support: number of examples (S) in the support set
        num_queries: number of examples (Q) in the query set
      Returns:
        ce_loss: the cross entropy loss between the predicted labels and true labels
        acc: the accuracy of classification on the queries
    """
    #############################
    #### YOUR CODE GOES HERE ####

    # compute the prototypes
    # print("in protoloss function")
    # print("x_latent_shape is ", x_latent)
    # print("q_latent shape is ", q_latent)
    # print("labels_onehot shape is ", labels_onehot.shape)
    # print("num_classes" ,num_classes)
    # print("num_support", num_support)
    # print("num_queries", num_queries)

    # print("reshaping labels_onehot")
    labels_onehot = tf.reshape(labels_onehot,[num_classes * num_queries, num_classes])
    # print("labels_onehot reshaped shape", labels_onehot.shape)
    x_latent = tf.reshape(x_latent, [num_classes, num_support, -1])
    prototypes = tf.reduce_mean(x_latent, 1)

    # print("after reshape x_latent.shape", x_latent.shape)
    # print("prototypes shape", prototypes.shape)

    # compute the distance from the prototypes
    # print("computing distances")
    # print("reshaping")
    prototypes = tf.reshape(prototypes, [1, num_classes, -1])
    q_latent = tf.reshape(q_latent, [num_queries * num_classes, 1, -1])
    # print("prototypes shape", prototypes.shape)
    # print("q_latent shape", q_latent)

    diff = prototypes - q_latent
    # print("diff shape", diff.shape)
    distances = tf.norm(diff, axis=2)
    # print("distances shape", distances.shape)
    logits = -distances
    # compute cross entropy loss
    # print("labels_onehot and distances should have same shape for cross entropy.")
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=-distances, labels=labels_onehot))

    # accuracy:
    # print("computing accuracy")
    predictions = tf.argmin(distances, axis=1)
    # print("predictions.shape", predictions)
    # print("predictions shape", predictions.shape)
    labels_argmaxed = tf.argmax(labels_onehot, 1)
    # print("labels_argmaxed.shape", labels_argmaxed.shape)
    # print("labels shape", labels.shape)
    labels_argmaxed = tf.reshape(labels_argmaxed, [-1])
    # print("reshaped labels_argmaxed.shape", labels_argmaxed.shape)
    # print("labels shape 2", labels.shape)
    acc = tf.reduce_mean(tf.cast(tf.equal(labels_argmaxed, predictions), dtype=tf.float32))
    # print(acc)
    #############################
    return ce_loss, acc