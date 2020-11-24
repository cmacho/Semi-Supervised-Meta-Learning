
import numpy as np
import random
import tensorflow as tf
import meta_data_loader as mdl
import os
import datetime
import matplotlib.pyplot as plt
import re
from functools import partial
import argparse
from models import ProtoNet, ProtoLoss

def parse_args():
    parser = argparse.ArgumentParser(description="tf implementation of MAML++")
    parser.add_argument('--val_data_path', type=str, default='/home/cle_macho/mini_imagenet/val', help = "path to validation dataset")
    parser.add_argument('--test_data_path', type=str, default='/home/cle_macho/mini_imagenet/test', help = "path to test dataset")
    parser.add_argument("--large_train_data_path", type=str, default='/home/cle_macho/mini_imagenet/train',
                        help="full training data for mini imagenet consisting of 64 classes")
    parser.add_argument("--small_train_data_path", type=str, default="/home/cle_macho/mini_imagenet/train_split/labeled_use",
                        help="the part of the training data split that is meant for labeled use, e.g. only 12 classes")
    parser.add_argument("--img_side_length", type = int, default=84, help="side length of input images")
    parser.add_argument("--channels", type=int, default=3, help="number of channels for input imgs")
    parser.add_argument("--num_tasks", type=int, default=100, help="how many tasks to sample when training with small num of tasks")
    parser.add_argument("--n_way", default=5, type=int, help="number of classes in each task")
    parser.add_argument("--k_shot", default=1, type =int, help="number of support images per class in a task")
    parser.add_argument("--num_query_per_class", default=5, type=int, help="number of query images per class in a task")
    parser.add_argument("--exp_string", default="exp1", type=str, help="name of subfolder in logdir where to save model checkpoints")
    parser.add_argument("--logdir", default="nonexistent_dir123123123", type=str, help="name of directory where to save data for experiment")
    parser.add_argument("--meta_batch_size", default=1, type=int, help="number of tasks per batch")
    parser.add_argument("--validation_iterations", default=100, type=int, help="number of tasks per batch for validation")
    parser.add_argument("--meta_lr", default=0.001, type=float, help="outer learning rate")
    parser.add_argument("--meta_train_iterations", default=60000, type=int, help="number of iterations for training")
    parser.add_argument("--test_iterations", default=1000, type=int, help="number of iterations for test")
    parser.add_argument("--log_file_name", default="params_run.txt", type=str, help="name of file that shows parameters of the experiments")
    parser.add_argument("--cosine_decay_alpha", default=0.001, type=float, help="alpha parameter for cosine lr schedule")
    parser.add_argument("--test_only", action="store_true", help="run only test, no training")
    parser.add_argument("--train_data_mode", type=str, choices=['small_number', 'full', 'unsupervised', 'mixed'], help="what kind of training data to load")
    parser.add_argument("--small_num_sampling_mode", type=str, choices=['fixed', 'new_sample'], default='fixed',
                        help="whether to sample new small number of tasks or load from file")
    parser.add_argument("--test_on_validation_set", action="store_true", help="test on validation set")
    parser.add_argument("--no_augmentation", action="store_true",
                        help="experiment without data augmentation. by default, data augmentation is used for "
                             "training with small")
    parser.add_argument('--number_labeled_tasks_per_batch', nargs="+", default=[1], type=int,
                        help="how many tasks in each batch should come from labeled data when train_data_mode is 'mixed'."
                             "This can be a list and the number of labeled tasks in consecutive batches will alternate between"
                             "the numbers in the list")
    parser.add_argument('--outer_lr_mode', type=str, choices=['fixed', 'cosine_decay'], default = 'fixed', help="whether to use lr schedule")
    parser.add_argument('--name_test_output', type=str, default='test_output', help="file name of test output")
    parser.add_argument('--file_name_images', type=str, default='images.npy', help="file of images to use for unsupervised / mixed mode")
    parser.add_argument('--file_name_cluster_labels', type=str, help="file of cluster labels to use for unsupervised / mixed mode")
    return parser.parse_args()


def main(args):
    args = parse_args()

    if not args.test_only:
        log_file_path = os.path.join(args.logdir, args.log_file_name)

        with open(log_file_path, "w") as f:
            datetime_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"current date and time is {datetime_string}", file=f)
            for arg in vars(args):
                print(f"{arg} = {getattr(args, arg)}", file=f)

    num_filters = [64] * 3
    latent_dim = 64
    model = ProtoNet(num_filters, latent_dim)

    if args.test_only:
        model_file = tf.train.latest_checkpoint(args.logdir + '/' + args.exp_string)
        print("Restoring model weights from ", model_file)
        model.load_weights(model_file)
    else:
        #train model
        print("creating data loader for train data")
        if args.train_data_mode == "small_number":
            if args.small_num_sampling_mode == 'new_sample':
                X, Y, total_num_classes = mdl.load_data_from_folders(args.small_train_data_path)
                initial_data_loader = mdl.DataLoader(X, Y, total_num_classes)
                tasks = initial_data_loader.sample_batch(args.n_way, args.k_shot + args.num_query_per_class, args.num_tasks)
            elif args.small_num_sampling_mode == 'fixed':
                expected_shape = (args.num_tasks, args.n_way, args.k_shot + args.num_query_per_class,
                                  args.img_side_length, args.img_side_length, args.channels)
                tasks = np.load("labeled_tasks.npy")
                if tasks.shape != expected_shape:
                    raise Exception(f"labeled_tasks.npy has shape + {tasks.shape}. Expected shape is {expected_shape}")
            else:
                raise Exception("no valid choice of args.small_num_sampling_mode")
            data_loader_train = mdl.DataLoaderFromSmallNumOfTasks(tasks)
        elif args.train_data_mode == "full":
            X, Y, total_num_classes = mdl.load_data_from_folders(args.large_train_data_path)
            data_loader_train = mdl.DataLoader(X, Y, total_num_classes)
        elif args.train_data_mode == "unsupervised":
            X = np.load(args.file_name_images)
            if args.file_name_cluster_labels is None:
                raise Exception("args.file_name_cluster_labels not specified")
            Y = np.load(args.file_name_cluster_labels)
            total_num_classes = [500] * 4
            data_loader_train = mdl.DataLoader(X, Y, total_num_classes)
        elif args.train_data_mode == "mixed":
            small_sample_list = args.number_labeled_tasks_per_batch
            unsupervised_sample_list = [1 - num for num in small_sample_list]

            if args.small_num_sampling_mode == 'new_sample':
                X, Y, total_num_classes = mdl.load_data_from_folders(args.small_train_data_path)
                initial_data_loader = mdl.DataLoader(X, Y, total_num_classes)

                tasks = initial_data_loader.sample_batch(args.n_way, args.k_shot + args.num_query_per_class,
                                                         args.num_tasks)
            elif args.small_num_sampling_mode == 'fixed':
                expected_shape = (args.num_tasks, args.n_way, args.k_shot + args.num_query_per_class,
                                  args.img_side_length, args.img_side_length, args.channels)
                tasks = np.load("labeled_tasks.npy")
                if tasks.shape != expected_shape:
                    raise Exception(f"labeled_tasks.npy has shape + {tasks.shape}. Expected shape is {expected_shape}")
            else:
                raise Exception("no valid choice of args.small_num_sampling_mode")
            data_loader_small = mdl.DataLoaderFromSmallNumOfTasks(tasks)

            X_unsup = np.load(args.file_name_images)
            if args.file_name_cluster_labels is None:
                raise Exception("args.file_name_cluster_labels not specified")
            Y_unsup = np.load(args.file_name_cluster_labels)
            total_num_classes = [500] * 4
            data_loader_unsupervised = mdl.DataLoader(X_unsup, Y_unsup, total_num_classes)

            data_loader_train = mdl.DataLoaderCombiner(data_loader_small, data_loader_unsupervised,
                                                       small_sample_list, unsupervised_sample_list)
        else:
            raise Exception("no valid train data mode specified.")
        
        print("creating data loader for validation data")
        # create data loader for val
        X_val, Y_val, total_num_classes_val = mdl.load_data_from_folders(args.val_data_path)
        data_loader_val = mdl.DataLoader(X_val,Y_val, total_num_classes_val)

        print("beginning training")
        # training
        augment = (args.train_data_mode == "small_number" or args.train_data_mode == "mixed"
                   or args.train_data_mode =="unsupervised") and not args.no_augmentation
        meta_train_fn(model, data_loader_train, data_loader_val, args, augment=augment)

    if args.test_on_validation_set:
        X_test, Y_test, total_num_classes_test = mdl.load_data_from_folders(args.val_data_path)
    else:
        X_test, Y_test, total_num_classes_test = mdl.load_data_from_folders(args.test_data_path)

    data_loader_test = mdl.DataLoader(X_test, Y_test, total_num_classes_test)

    meta_test_fn(model, data_loader_test, args)


def create_random_labels(n_way, num_per_class, meta_batch_size):
    # create random labels.
    label_tr_np = np.zeros((meta_batch_size, n_way, num_per_class, n_way))
    for ii in range(meta_batch_size):
        shuffled_identity_matrix = np.eye(n_way)
        np.random.shuffle(shuffled_identity_matrix)

        label_tr_np[ii, :, :, :] = shuffled_identity_matrix.reshape((n_way, 1, n_way))

    label_tr = tf.constant(label_tr_np)
    return label_tr


def load_batch_and_labels(data_loader, n_way, k_shot, num_query_per_class, meta_batch_size, augment = True, shuffle = True, flip = True):
    all_images_batch = data_loader.sample_batch(num_classes=n_way,
                                                samples_per_class=k_shot + num_query_per_class,
                                                batch_size=meta_batch_size)
    if augment:
        all_images_batch = mdl.augment_data(all_images_batch, shuffle=shuffle, flip=flip)
    input_tr = all_images_batch[:, :, :k_shot, :, :, :]
    input_ts = all_images_batch[:, :, k_shot:, :, :, :]
    label_all = create_random_labels(n_way, k_shot + num_query_per_class, meta_batch_size)
    label_tr = label_all[:, :, :k_shot, :]
    label_ts = label_all[:, :, k_shot:, :]


    # reshape everything, reducing the num_classes and samples_per_class dimensions to one
    input_tr = input_tr.reshape(
        (input_tr.shape[0], input_tr.shape[1] * input_tr.shape[2], input_tr.shape[3], input_tr.shape[4],
         input_tr.shape[5]))
    input_ts = input_ts.reshape(
        (input_ts.shape[0], input_ts.shape[1] * input_ts.shape[2], input_ts.shape[3], input_ts.shape[4],
         input_ts.shape[5]))
    label_tr = tf.reshape(label_tr, [meta_batch_size, n_way * k_shot, n_way])
    label_ts = tf.reshape(label_ts, [meta_batch_size, n_way * num_query_per_class, n_way])
    return input_tr, input_ts, label_tr, label_ts


def meta_train_fn(model, data_loader_train, data_loader_val, args, augment):
    SUMMARY_INTERVAL = 100
    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 10

    post_accuracies = []
    val_accuracies = []
    train_accs_smoothed = []

    # initialize
    acc_smoothed = 1. / args.n_way

    if args.outer_lr_mode == 'cosine_decay':
        scheduler = tf.keras.experimental.CosineDecay(initial_learning_rate = args.meta_lr,
                                                  decay_steps = args.meta_train_iterations, alpha = args.cosine_decay_alpha)
        optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)
    elif args.outer_lr_mode == 'fixed':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.meta_lr)
    else:
        raise Exception("no valid choice of args.outer_lr_mode")

    for itr in range(args.meta_train_iterations):
        gradient_list = []
        for jj in range(args.meta_batch_size):
            inp = load_batch_and_labels(data_loader_train, args.n_way, args.k_shot, args.num_query_per_class,
                                        1, augment=augment)
            input_tr, input_ts, _, _ = inp


            # make train step
            label_ts = np.zeros((1, args.n_way, args.num_query_per_class, args.n_way))
            identity_matrix = np.eye(args.n_way)
            label_ts = label_ts + identity_matrix.reshape((1, args.n_way, 1, args.n_way))

            # print("label_ts.shape is", label_ts.shape)
            # print("label_ts is ", label_ts)

            input_tr = tf.reshape(input_tr, [-1, args.img_side_length, args.img_side_length, args.channels])
            input_ts = tf.reshape(input_ts, [-1, args.img_side_length, args.img_side_length, args.channels])

            # print("AFTER reshape:")
            # print("input_tr_shape is", input_tr.shape)
            # print("input_ts shape is", input_ts.shape)

            with tf.GradientTape() as tape:
                x_latent = model(input_tr)
                q_latent = model(input_ts)
                # print("x_latent shape is ", x_latent.shape)
                # print("q_latent shape is ", q_latent.shape)

                # print("computing proto loss:")
                ce_loss, acc = ProtoLoss(x_latent, q_latent, label_ts, args.n_way, args.k_shot, args.num_query_per_class)

            current_gradients = tape.gradient(ce_loss, model.trainable_variables)
            gradient_list.append(current_gradients)

        assert len(gradient_list) == args.meta_batch_size
        gradients = [tf.reduce_sum([g[ii] for g in gradient_list], axis=0) for ii in range(len(model.trainable_variables))]
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if itr % SUMMARY_INTERVAL == 0:
            acc_smoothed = 0.75 * acc_smoothed + 0.25 * acc
            train_accs_smoothed.append((itr, acc_smoothed))

        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            print(f'Iteration {itr} train accuracy: {acc}')

        if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
            acc_list1 = []
            for ii in range(args.validation_iterations):
                inp = load_batch_and_labels(data_loader_val, args.n_way, args.k_shot,
                                            args.num_query_per_class, 1, augment=False)
                input_tr, input_ts, label_tr, label_ts = inp

                label_ts = np.zeros((1, args.n_way, args.num_query_per_class, args.n_way))
                identity_matrix = np.eye(args.n_way)
                label_ts = label_ts + identity_matrix.reshape((1, args.n_way, 1, args.n_way))

                input_tr = tf.reshape(input_tr, [-1, args.img_side_length, args.img_side_length, args.channels])
                input_ts = tf.reshape(input_ts, [-1, args.img_side_length, args.img_side_length, args.channels])

                x_latent = model(input_tr, training=False)
                q_latent = model(input_ts, training=False)
                ce_loss, acc_val = ProtoLoss(x_latent, q_latent, label_ts, args.n_way, args.k_shot,
                                         args.num_query_per_class)

                acc_list1.append(acc_val)

            acc_val_total = tf.reduce_mean(acc_list1)
            print(f'Validation accuracy: {acc_val_total}')


            val_accuracies.append((itr, acc_val_total))

    model_file = args.logdir + '/' + args.exp_string + '/model'
    print("Saving to ", model_file)
    model.save_weights(model_file)

    #save plot
    plot_path = os.path.join(args.logdir, "accuracy_plot.png")
    plt.plot([a[0] for a in train_accs_smoothed], [a[1] for a in train_accs_smoothed], label='train accuracy smoothed')
    plt.plot([a[0] for a in val_accuracies], [a[1] for a in val_accuracies], label='validation accuracy')
    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(plot_path)

    #save_data as csv
    train_acc_csv_path = os.path.join(args.logdir, "train_accuracies.csv")
    val_acc_csv_path = os.path.join(args.logdir, "val_accuracies.csv")
    with open(train_acc_csv_path, 'w') as f:
        print("Iteration, Accuracy", file=f)
        for tup in post_accuracies:
            print(f"{tup[0]}, {tup[1]}", file=f)
    with open(val_acc_csv_path, 'w') as f:
        print("Iteration, Accuracy", file=f)
        for tup in val_accuracies:
            print(f"{tup[0]}, {tup[1]}", file=f)

def meta_test_fn(model, data_loader_test, args):

    test_accuracies = []
    test_accuracies_sum = 0
    for ii in range(args.test_iterations):
        print(f"test batch {ii} of {args.test_iterations}")
        inp = load_batch_and_labels(data_loader_test, args.n_way, args.k_shot,
                                    args.num_query_per_class, 1, augment=False)
        input_tr, input_ts, label_tr, label_ts = inp
        label_ts = np.zeros((1, args.n_way, args.num_query_per_class, args.n_way))
        identity_matrix = np.eye(args.n_way)
        label_ts = label_ts + identity_matrix.reshape((1, args.n_way, 1, args.n_way))

        input_tr = tf.reshape(input_tr, [-1, args.img_side_length, args.img_side_length, args.channels])
        input_ts = tf.reshape(input_ts, [-1, args.img_side_length, args.img_side_length, args.channels])

        x_latent = model(input_tr, training=False)
        q_latent = model(input_ts, training=False)
        ce_loss, acc_test = ProtoLoss(x_latent, q_latent, label_ts, args.n_way, args.k_shot,
                                      args.num_query_per_class)
        test_accuracies.append(acc_test)
        test_accuracies_sum = test_accuracies_sum + acc_test
    total_test_accuracy = test_accuracies_sum / args.test_iterations

    print("test accuracy is", total_test_accuracy.numpy())

    log_file_path = os.path.join(args.logdir, args.name_test_output)
    with open(log_file_path, "w") as f:
        datetime_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"current date and time is {datetime_string}", file=f)
        for arg in vars(args):
            print(f"{arg} = {getattr(args, arg)}", file=f)
        print(f"test accuracy is {total_test_accuracy}", file=f)


if __name__ == '__main__':
    args = parse_args()
    main(args)
