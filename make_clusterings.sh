#!/bin/bash

mkdir -p cluster_labels_20_partitions
python kmeans_on_embeddings.py --clustering kmeans
mv cluster_labels.npy cluster_labels_20_partitions

rm cluster_labels.npy
mkdir -p cluster_labels_20_partitions_constrained_standard
python kmeans_on_embeddings.py --clustering kmeans --file_name_embedding embedding_unlabeled.npy
mv cluster_labels.npy cluster_labels_20_partitions_constrained_standard

rm cluster_labels.npy
mkdir -p cluster_labels_20_partitions_constrained_constrained
python kmeans_on_embeddings.py --clustering constrained_kmeans
mv cluster_labels.npy cluster_labels_20_partitions_constrained_constrained

rm cluster_labels.npy
mkdir -p cluster_labels_20_partitions_standard_then_constrained
python kmeans_on_embeddings.py --clustering constrained_kmeans \
 --file_name_embedding_unlabeled embedding_standard_unlabeled.npy \
 --file_name_embedding_labeled embedding_standard_labeled.npy
mv cluster_labels.npy cluster_labels_20_partitions_standard_then_constrained
