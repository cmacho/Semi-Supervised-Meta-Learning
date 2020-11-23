
import numpy as np
import random
import tensorflow as tf
from MAML_plus_plus import MAMLpp
from MAML import MAML
import meta_data_loader as mdl
import os
import datetime
import matplotlib.pyplot as plt
import re
from functools import partial

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="tf implementation of MAML++")
    parser.add_argument('--val_data_path', type=str, default='/home/cle_macho/mini_imagenet/val', help = "path to validation dataset")
    parser.add_argument('--test_data_path', type=str, default='/home/cle_macho/mini_imagenet/test', help = "path to test dataset")
    parser.add_argument("--large_train_data_path", type=str, default='/home/cle_macho/mini_imagenet/train',
                        help="full training data for mini imagenet consisting of 64 classes")
    parser.add_argument("--small_train_data_path", type=str, default="/home/cle_macho/mini_imagenet/train_split/labeled_use",
                        help="the part of the training data split that is meant for labeled use, e.g. only 12 classes")
    parser.add_argument('--num_inner_updates', type=int, default=5, help="number of gradient descent steps in inner loop of maml")
    parser.add_argument("--img_side_length", type = int, default=84, help="side length of input images")
    parser.add_argument("--channels", type=int, default=3, help="number of channels for input imgs")
    parser.add_argument("--num_tasks", type=int, default=100, help="how many tasks to sample when training with small num of tasks")
    parser.add_argument("--n_way", default=5, type=int, help="number of classes in each task")
    parser.add_argument("--k_shot", default=1, type =int, help="number of support images per class in a task")
    parser.add_argument("--num_query_per_class", default=5, type=int, help="number of query images per class in a task")
    parser.add_argument("--exp_string", default="exp1", type=str, help="name of subfolder in logdir where to save model checkpoints")
    parser.add_argument("--logdir", default="nonexistent_dir123123123", type=str, help="name of directory where to save data for experiment")
    parser.add_argument("--meta_batch_size", default=4, type=int, help="number of tasks per batch")
    parser.add_argument("--validation_batch_size", default=32, type=int, help="number of tasks per batch for validation")
    parser.add_argument("--meta_lr", default=0.005, type=float, help="outer learning rate")
    parser.add_argument("--inner_update_lr", default=0.25, type=float, help="initialization for inner learning rates")
    parser.add_argument("--meta_train_iterations", default=60000, type=int, help="number of iterations for training")
    parser.add_argument("--test_batch_size", default=8, type=int, help="number of tasks per batch for test")
    parser.add_argument("--test_iterations", default=125, type=int, help="number of iterations for test")
    parser.add_argument("--max_iter_multi_loss", default=7500, type=int, help="how many iterations to use multi loss for")
    parser.add_argument("--num_first_order_iters", default=0, type=int, help="how many iterations to use first order maml for")
    parser.add_argument("--log_file_name", default="params_run.txt", type=str, help="name of file that shows parameters of the experiments")
    parser.add_argument("--cosine_decay_alpha", default=0.001, type=float, help="alpha parameter for cosine lr schedule")
    parser.add_argument("--test_only", action="store_true", help="run only test, no training")
    parser.add_argument("--train_data_mode", type=str, choices=['small_number', 'full', 'unsupervised', 'mixed'], help="run only test, no training")
    parser.add_argument("--small_num_sampling_mode", type=str, choices=['fixed', 'new_sample'], default='fixed',
                        help="whether to sample new small number of tasks or load from file")
    parser.add_argument("--test_on_validation_set", action="store_true", help="test on validation set")
    parser.add_argument("--multiply_by_five", action="store_true", help="during testing, multiply the learned lr of "
                                                                        "mamlpp by five before applying it (for "
                                                                        "compatibility with an older version of "
                                                                        "the mamlpp model)")
    parser.add_argument("--no_augmentation", action="store_true",
                        help="experiment without data augmentation. by default, data augmentation is used for "
                             "training with small")
    parser.add_argument('--number_labeled_tasks_per_batch', nargs="+", default=[1], type=int,
                        help="how many tasks in each batch should come from labeled data when train_data_mode is 'mixed'."
                             "This can be a list and the number of labeled tasks in consecutive batches will alternate between"
                             "the numbers in the list")
    parser.add_argument('--model', type=str, choices=['mamlpp', 'maml'], default='maml', help="maml++ or maml")
    parser.add_argument('--outer_lr_mode', type=str, choices=['fixed', 'cosine_decay'], default='fixed', help="whether to use lr schedule")
    parser.add_argument('--val_inner_updates', type=int, default=5, help="number inner updates for validation")
    parser.add_argument('--test_inner_updates', type=int, default=50, help="number inner updates for test")
    parser.add_argument('--name_test_output', type=str, default='test_output', help="file name of test output")
    parser.add_argument('--file_name_images', type=str, default='images.npy', help="file of images to use for unsupervised / mixed mode")
    parser.add_argument('--file_name_cluster_labels', type=str, help="file of cluster labels to use for unsupervised / mixed mode")
    parser.add_argument("--no_multi_loss", action="store_true", help="use standard loss for maml instead of the multi loss from maml++")
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
    if args.model == 'mamlpp':
        model = MAMLpp(args.img_side_length, args.channels, dim_output=args.n_way,
             inner_update_lr=args.inner_update_lr, num_filters=32, 
             num_inner_updates = args.num_inner_updates)
    elif args.model == 'maml':
        model = MAML(args.img_side_length, args.channels, dim_output=args.n_way,
             inner_update_lr=args.inner_update_lr, num_filters=32,
             num_inner_updates = args.num_inner_updates)
    else:
        raise Exception("no valid choice of args.model")

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
            unsupervised_sample_list = [args.meta_batch_size - num for num in small_sample_list]

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
                   or args.train_data_mode == "unsupervised") and not args.no_augmentation
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

@tf.function
def single_task_first_order_outer_train_step(inp, model, loss_weights, index_of_weight, model_type):
    input_tr, input_ts, label_tr, label_ts = inp
    weights = model.conv_layers.conv_weights
    ts_grad_list = []
    outputs_list = []
    num_inner_updates = model.num_inner_updates

    for ii in range(num_inner_updates):
        # print("inner loop input_tr shape", input_tr.shape)
        with tf.GradientTape(persistent=False) as inner_tape:
            inner_tape.watch(weights)
            if model_type == 'maml':
                outputs = model.conv_layers(input_tr, weights)
            else:
                outputs = model.conv_layers(input_tr, weights, ii)
            loss_tr = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=label_tr))
        inner_grad = inner_tape.gradient(loss_tr, weights)

        variable_list = []
        for var in model.trainable_variables:
            variable_list.append(var)
        for w in weights:
            variable_list[index_of_weight[w]] = weights[w]

        with tf.GradientTape(persistent=False) as ts_tape:
            ts_tape.watch(variable_list)
            if model_type == 'mamlpp':
                new_weights = {w_name: w - model.inner_update_lr_dict[w_name][ii] * inner_grad[w_name] for w_name, w in
                   weights.items()}
            else:
                new_weights = {w_name: w - model.inner_update_lr * inner_grad[w_name] for w_name, w in
                   weights.items()}
            if model_type == 'maml':
                output_ts = model.conv_layers(input_ts, new_weights)
            else:
                output_ts = model.conv_layers(input_ts, new_weights, ii)
            loss_ts = loss_weights[ii] * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_ts, labels=label_ts))
        ts_grad = ts_tape.gradient(loss_ts, variable_list)
        ts_grad_list.append(ts_grad)
        outputs_list.append(output_ts)
        weights = new_weights

    for i, var in enumerate(model.trainable_variables):
        for g in ts_grad_list:
            if g[i] is None:
                g[i] = 0 * var

    gradients = [tf.reduce_sum([g[ii] for g in ts_grad_list], axis=0) for ii in range(len(model.trainable_variables))]
    return [outputs_list[-1], gradients]  # we only return the output after the last update step


@tf.function
def outer_train_step_first_order(inp, model, optim, loss_weights, index_of_weight, model_type):
    input_tr, input_ts, label_tr, label_ts = inp
    output =[]
    batch_gradients = []
    num_inner_updates = loss_weights.shape[0]
    batch_size = input_tr.shape[0]

    single_task_partial = partial(single_task_first_order_outer_train_step, model=model, loss_weights=loss_weights,
                                  index_of_weight=index_of_weight, model_type=model_type)
    out_dtype = [tf.float32, [tf.float32] * len(model.trainable_variables)]
    result = tf.map_fn(single_task_partial,
                       elems=(input_tr, input_ts, label_tr, label_ts),
                       dtype=out_dtype,
                       parallel_iterations=8)
    output = result[0]
    batch_gradients = result[1]
    total_grad = [tf.reduce_mean(g, axis=0) for g in batch_gradients]
    optim.apply_gradients(zip(total_grad, model.trainable_variables))
    return output


@tf.function
def outer_train_step(inp, model, optim, loss_weights):
    input_tr, input_ts, label_tr, label_ts = inp

    num_inner_updates = loss_weights.shape[0]

    with tf.GradientTape(persistent=False) as outer_tape:
        output = model(inp)

        total_loss = tf.constant(0, dtype=tf.float32)

        for ii in range(num_inner_updates):
            loss_ts = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output[ii], labels=label_ts))
            total_loss = total_loss + loss_weights[ii] * loss_ts

    gradients = outer_tape.gradient(total_loss, model.trainable_variables)
    gradients = [tf.clip_by_value(g, clip_value_min=-10, clip_value_max=10) for g in gradients]
    optim.apply_gradients(zip(gradients, model.trainable_variables))
    return output[num_inner_updates-1]

def meta_train_fn(model, data_loader_train, data_loader_val, args, augment):
    SUMMARY_INTERVAL = 10
    SAVE_INTERVAL = 100
    PRINT_INTERVAL = 10
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 10

    post_accuracies = []
    val_accuracies = []
    train_accs_smoothed = []
    min_loss_weights = 0.03 / args.num_inner_updates

    #initialize
    acc_smoothed = 1. / args.n_way

    if args.outer_lr_mode == 'cosine_decay':
        scheduler = tf.keras.experimental.CosineDecay(initial_learning_rate = args.meta_lr,
                                                  decay_steps = args.meta_train_iterations, alpha = args.cosine_decay_alpha)
        optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)
    elif args.outer_lr_mode == 'fixed':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.meta_lr)
    else:
        raise Exception("no valid choice of args.outer_lr_mode")

    weights = model.conv_layers.conv_weights
    index_of_weight = {}
    for w in weights:
        pattern = re.compile(w)
        count = 0
        for i, var in enumerate(model.trainable_variables):
            if pattern.match(var.name) is not None:
                index_of_weight[w] = i
                count = count + 1
        if count != 1:
            raise Exception("not exactly one match in trainable variables found for name of weight")


    for itr in range(args.meta_train_iterations):
        inp = load_batch_and_labels(data_loader_train, args.n_way, args.k_shot, args.num_query_per_class,
                                    args.meta_batch_size, augment=augment)

        input_tr, input_ts, label_tr, label_ts = inp
        
        loss_weights = np.ones((args.num_inner_updates,))
        if args.no_multi_loss:
            loss_weights[:args.num_inner_updates - 1] = 0.
        else:
            for ii in range(args.num_inner_updates - 1):
                loss_weights[ii] = max(1./args.num_inner_updates - itr / args.max_iter_multi_loss * (1./args.num_inner_updates - min_loss_weights),
                                              min_loss_weights)
            loss_weights[args.num_inner_updates - 1] = min(1./args.num_inner_updates + (args.num_inner_updates-1)
                                                               * itr / args.max_iter_multi_loss
                                                               * (1./args.num_inner_updates - min_loss_weights),
                                                               1. - (args.num_inner_updates-1) * min_loss_weights)
        loss_weights = tf.constant(loss_weights, dtype=tf.float32)

        if itr == 0:
            # initialize batch norm:
            for ii in range(args.num_inner_updates):
                if args.model == 'maml':
                    unused = model.conv_layers(input_tr, model.conv_layers.conv_weights)
                else:
                    unused = model.conv_layers(input_tr, model.conv_layers.conv_weights, ii)

        if itr < args.num_first_order_iters:
            output = outer_train_step_first_order(inp, model, optimizer, loss_weights, index_of_weight, args.model)
            output = tf.stack(output)
        else:
            output = outer_train_step(inp, model, optimizer, loss_weights)


        if itr % SUMMARY_INTERVAL == 0:
            predictions = tf.argmax(output, axis=-1)
            label_ts_argmaxed = tf.argmax(label_ts, axis=-1)

            acc = tf.reduce_mean(tf.cast(tf.equal(label_ts_argmaxed, predictions), dtype=tf.float32))
            post_accuracies.append((itr, acc))
            acc_smoothed = 0.75 * acc_smoothed + 0.25 * acc
            train_accs_smoothed.append((itr, acc_smoothed))

        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            print(f'Iteration {itr} post-inner-loop test accuracy: {post_accuracies[-1][1]}')

        if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
            inp = load_batch_and_labels(data_loader_val, args.n_way, args.k_shot,
                                        args.num_query_per_class, args.validation_batch_size, augment=False)
            input_tr, input_ts, label_tr, label_ts = inp

            if args.model == 'mamlpp':
                val_out_all = model(inp, training=False)
            else:
                val_out_all = model(inp, num_inner_updates=args.val_inner_updates, training=False)
            val_out = val_out_all[args.num_inner_updates-1]
            predictions = tf.argmax(val_out, axis=-1)
            label_ts_argmaxed = tf.argmax(label_ts, axis=-1)
            acc_val = tf.reduce_mean(tf.cast(tf.equal(label_ts_argmaxed, predictions), dtype=tf.float32))

            print(f'Meta validation post-inner-loop test accuracy: {acc_val}')
            print("loss weights is " , loss_weights)

            val_accuracies.append((itr, acc_val))

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

@tf.function
def call_model_for_test(model, inp, num_inner_updates):
    test_out_all = model(inp, num_inner_updates=num_inner_updates, training=False)
    return test_out_all

def meta_test_fn(model, data_loader_test, args):

    if args.model == "mamlpp":
        for w in model.inner_update_lr_dict:
            for ii in range(model.num_inner_updates):
                print(w, ii, model.inner_update_lr_dict[w][ii])

    test_accuracies = []
    test_accuracies_sum = 0
    for ii in range(args.test_iterations):
        print(f"test batch {ii} of {args.test_iterations}")
        inp = load_batch_and_labels(data_loader_test, args.n_way, args.k_shot,
                                    args.num_query_per_class, args.test_batch_size, augment=False)
        input_tr, input_ts, label_tr, label_ts = inp
        if args.model == 'mamlpp':
            test_out_all = model(inp, multiply_by_five = args.multiply_by_five, training = False)
        else:
            test_out_all = call_model_for_test(model, inp, num_inner_updates=args.test_inner_updates)
        test_out = test_out_all[-1]

        predictions = tf.argmax(test_out, axis=-1)
        label_ts_argmaxed = tf.argmax(label_ts, axis=-1)

        acc_test = tf.reduce_mean(tf.cast(tf.equal(label_ts_argmaxed, predictions), dtype=tf.float32))
        test_accuracies.append(acc_test)
        test_accuracies_sum = test_accuracies_sum + acc_test
    total_test_accuracy = test_accuracies_sum / args.test_iterations

    print("test accuracy is", total_test_accuracy.numpy())

    log_file_path = os.path.join(args.logdir, args.name_test_output)
    with open(log_file_path,"w") as f:
        datetime_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"current date and time is {datetime_string}", file=f)
        for arg in vars(args):
            print(f"{arg} = {getattr(args, arg)}", file=f)
        print(f"test accuracy is {total_test_accuracy}", file=f)


if __name__ == '__main__':
    args = parse_args()
    main(args)
