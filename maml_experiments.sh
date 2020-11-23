#!/bin/bash

LABELS="cluster_labels_20_partitions/cluster_labels.npy"

FOLDER="maml_unsupervised"

mkdir -p ${FOLDER}

python train_MAML.py --logdir ${FOLDER} --train_data_mode unsupervised \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 5

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 20

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 50

FOLDER = "maml_mixed"

mkdir -p ${FOLDER}

python train_MAML.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 2
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 5

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 20

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 50

FOLDER="maml_supervised"

mkdir -p ${FOLDER}

python train_MAML.py --logdir ${FOLDER} --train_data_mode small_number \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 5

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 20

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 50


LABELS="cluster_labels_20_partitions_constrained_constrained/cluster_labels.npy"
FOLDER="maml_constrained_constrained_mixed"

mkdir -p ${FOLDER}

python train_MAML.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 2
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 5

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 20

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 50


LABELS="cluster_labels_20_partitions_constrained_standard/cluster_labels.npy"

FOLDER="maml_constrained_standard_mixed"

mkdir -p ${FOLDER}

python train_MAML.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 2
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 5

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 20

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 50

LABELS="cluster_labels_20_partitions_standard_then_constrained/cluster_labels.npy"

FOLDER="maml_standard_then_constrained_mixed"

mkdir -p ${FOLDER}

python train_MAML.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 2
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 5

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 20

python train_MAML.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_5_shot --k_shot 50
