#!/bin/bash

LABELS="cluster_labels_20_partitions/cluster_labels.npy"

FOLDER="proto_unsupervised"

python train_proto.py --logdir ${FOLDER} --train_data_mode unsupervised \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_supervised"

python train_proto.py --logdir ${FOLDER} --train_data_mode small_number \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_mixed_0_0_0_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 0 0 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_mixed_0_0_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 0 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_mixed_0_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_mixed_0_1_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 1 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_mixed_0_1_1_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 1 1 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50


LABELS="cluster_labels_20_partitions_constrained_constrained/cluster_labels.npy"

FOLDER="proto_constrained_constrained_mixed_0_0_0_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 0 0 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50


FOLDER="proto_constrained_constrained_mixed_0_0_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 0 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_constrained_constrained_mixed_0_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50


FOLDER="proto_constrained_constrained_mixed_0_1_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 1 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_constrained_constrained_mixed_0_1_1_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 1 1 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50


LABELS="cluster_labels_20_partitions_constrained_standard/cluster_labels.npy"

FOLDER="proto_constrained_standard_mixed_0_0_0_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 0 0 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_constrained_standard_mixed_0_0_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 0 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_constrained_standard_mixed_0_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_constrained_standard_mixed_0_1_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 1 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_constrained_standard_mixed_0_1_1_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 1 1 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

LABELS="cluster_labels_20_partitions_standard_then_constrained/cluster_labels.npy"

FOLDER="proto_standard_then_constrained_mixed_0_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_standard_then_constrained_mixed_0_0_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 0 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_standard_then_constrained_mixed_0_1_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 1 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_standard_then_constrained_mixed_0_0_0_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 0 0 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50

FOLDER="proto_standard_then_constrained_mixed_0_1_1_1"

mkdir -p ${FOLDER}

python train_proto.py --logdir ${FOLDER} --train_data_mode mixed \
--number_labeled_tasks_per_batch 0 1 1 1 \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS}  \
--name_test_output test_output_1_shot --test_iterations 5000

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} --test_on_validation_set  \
--name_test_output val_output_50_shot --test_iterations 1000 --k_shot 50

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_5_shot --test_iterations 1000 --k_shot 5

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_20_shot --test_iterations 1000 --k_shot 20

python train_proto.py --logdir ${FOLDER} --test_only \
--file_name_cluster_labels ${LABELS} \
--name_test_output test_output_50_shot --test_iterations 1000 --k_shot 50