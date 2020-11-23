
from sklearn.cluster import KMeans
import numpy as np
import argparse
from constrained_kmeans import cluster_with_constraints

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
    parser.add_argument("--num_partitions", type=int, default = 20, help="number partitions to create")
    parser.add_argument("--num_without_rescaling", type=int, default=10, help="number partitions to create")
    parser.add_argument("--n_clusters", type=int, default = 500, help="number clusters")
    parser.add_argument("--n_init", type=int, default = 3, help="how often to run kmeans with different initializations to create each partition")
    parser.add_argument("--num_samples_per_class", type=int, default = 6, help="minimum number of points we would like to have in each class")
    parser.add_argument('--num_centers_from_labeled_tasks', nargs="+", default=[10] * 20, type=int,
                        help="how many centers are initialized from labeled tasks for each of the n_init")
    parser.add_argument("--clustering", type=str, choices=['kmeans', 'constrained_kmeans'], help="which clustering algorithm to use")
    parser.add_argument("--file_name_embedding", type=str, default="embedding.npy", help="file name for embedding that"
                                                                                         "is used for (unconstrained) kmeans")
    parser.add_argument("--file_name_embedding_unlabeled", type=str, default="embedding_unlabeled.npy", help="file name for embedding "
                                                                                                             "of unlabeled images that"
                                                                                                             "is used for constrained kmeans")
    parser.add_argument("--file_name_embedding_labeled", type=str, default="embedding_labeled.npy", help="file name for embedding "
                                                                                                             "of labeled imagesthat"
                                                                                                             "is used for constrained kmeans")
    return parser.parse_args()


args = parse_args()

if args.clustering == "kmeans":
    embedding = np.load(args.file_name_embedding)
elif args.clustering == "constrained_kmeans":
    embedding_labeled = np.load(args.file_name_embedding_labeled)
    embedding_unlabeled = np.load(args.file_name_embedding_unlabeled)
    print("shapes labeled unlabeled")
    print(embedding_labeled.shape)
    print(embedding_unlabeled.shape)
else:
    raise Exception("illegal choice of args.clustering?")

n_clusters = args.n_clusters
n_init = args.n_init
num_samples_per_class = args.num_samples_per_class
num_centers_from_labeled_tasks = args.num_centers_from_labeled_tasks
num_partitions = args.num_partitions


k_means_list = []
for i in range(num_partitions):
    print(f"i is {i}")
    while True:
        if args.clustering == "kmeans":
            if i < args.num_without_rescaling:
                encoding = embedding
            else:
                weight_vector = np.random.uniform(low=0.0, high=1.0, size=(1, embedding.shape[1]))
                encoding = weight_vector * embedding
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=n_init, max_iter=3000).fit(encoding)
            labels = kmeans.labels_
        elif args.clustering == "constrained_kmeans":
            if i < args.num_without_rescaling:
                encoding_labeled = embedding_labeled
                encoding_unlabeled = embedding_unlabeled
            else:
                weight_vector = np.random.uniform(low=0.0, high=1.0, size=(1, embedding_unlabeled.shape[1]))
                encoding_labeled = weight_vector * embedding_labeled
                encoding_unlabeled = weight_vector * embedding_unlabeled
            for j in range(n_init):
                if j == 0:
                    assignments_constrained_array, assignments_unconstrained_array, loss = cluster_with_constraints(encoding_labeled,encoding_unlabeled,
                                                                                                                    num_centers_from_labeled_tasks[i], n_clusters, max_iter=500)
                else:
                    new_constrained, new_unconstrained, new_loss = cluster_with_constraints(encoding_labeled,encoding_unlabeled,
                                                                                            num_centers_from_labeled_tasks[i], n_clusters, max_iter=500)
                    if new_loss < loss:
                        assignments_constrained_array, assignments_unconstrained_array, loss = new_constrained, new_unconstrained, new_loss
            labels = assignments_unconstrained_array
        else:
            raise Exception("illegal choice of args.clustering?")

        uniques, counts = np.unique(labels, return_counts=True)
        num_big_enough_clusters = np.sum(counts > num_samples_per_class)
        print(f"num_big_enough_clusters is {num_big_enough_clusters}")
        if num_big_enough_clusters > 0.9 * n_clusters:
            break
        else:
            print(f"repeating. num_big_enough_clusters is only {num_big_enough_clusters}.")
            print(counts)
    k_means_list.append(labels)

all_clusterings = np.array(k_means_list)

np.save("cluster_labels.npy", all_clusterings)

