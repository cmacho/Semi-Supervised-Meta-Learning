# Semi-Supervised Meta-Learning

This repository contains the code for my project on Semi-Supervised Meta Learning for the Stanford class CS330: Deep Multi-Task and Meta Learning.

The projet report, containing a detailed description of the project is [here](report).

The implementation of Constrained DeepCluster is here: [Constrained DeepCluster](https://github.com/cmacho/deepcluster)

# Usage

In order to run the experiments as described in the project report, go through the following steps:

1. Follow the instructions in the [Constrained DeepCluster](https://github.com/cmacho/deepcluster) repository to download and prepare the mini Imagenet data set and run both DeepCluster and Constrained DeepCluster.
2. The following files are created as a result of running DeepCluster and Constrained DeepCluster:  `labeled_tasks.npy`, `embedding.npy`, `images.npy`, `embedding_standard_labeled.npy`,`embedding_standard_unlabeled.npy`,`embedding_labeled.npy`, `embedding_unlabeled.npy`, `images_unlabeled.npy`.  Copy these files over here into this directory.
3. Run `make_clusterings.sh` in order to create partitions by using k-means and constrained k-means on partitions.
4. Run `proto_experiments.sh` in order to run the experiments with ProtoNets and/or run `maml_experiments.sh` in order to run the experiments with MAML.

# Requirements
- Python 3.7
- tensorflow 2
- scikit-learn 0.23.2
