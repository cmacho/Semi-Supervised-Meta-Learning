

import numpy as np
import random
import time
from sklearn.metrics import pairwise_distances

def sample_index_according_to_dist(weights):
    """
    weights: numpy array of shape (length,)
    returns: an index in range(length) sampled with probabilities proportional to weights
    """
    total = np.sum(weights)
    assert total > 0
    random_cutoff = random.random() * total
    running_sum = 0
    for j in range(weights.shape[0]):
        running_sum += weights[j]
        if running_sum > random_cutoff:
            return j
    raise Exception("should not reach here.")

def cluster_with_constraints(data_constrained, data_unconstrained, num_centers_from_labeled_tasks, n_clusters, max_iter):
    """ clustering """
    start_time = time.time()
    num_tasks, n_way, num_samples_per_class, dim = data_constrained.shape
    assert data_unconstrained.shape[1] == dim

    # collapse examples from same class in each task to their average
    data_constrained_concentrated = np.mean(data_constrained, axis=2)

    # ########################
    # initial centroids
    # ########################
    print("initializing centroids")
    cluster_centers = []

    # pick one task at random
    first_task_idx = random.choice(range(num_tasks))
    for i in range(n_way):
        cluster_centers.append(data_constrained_concentrated[first_task_idx, i])
        assert cluster_centers[-1].shape == (dim,)

    print("initializing centroids from labeled tasks")
    # pick further centers from data_constrained_concentrated until num_centers_from_labeled_tasks are picked.
    data_constrained_flat = data_constrained_concentrated.reshape((num_tasks * n_way, dim))
    for i in range(num_centers_from_labeled_tasks - n_way):
        #print(f"picking {i+n_way+1}th centroid")
        centers_array = np.array(cluster_centers)
        data_constrained_flat_rshp = data_constrained_flat.reshape((num_tasks * n_way, 1, dim))
        centers_array_rshp = centers_array.reshape((1, len(centers_array), dim))
        differences = data_constrained_flat_rshp - centers_array_rshp
        distances_squared = np.sum(differences**2, axis=2)
        D = np.min(distances_squared, axis=1)
        #print("D:")
        #print(D)
        j = sample_index_according_to_dist(D)
        cluster_centers.append(data_constrained_flat[j])

    assert(len(cluster_centers) == num_centers_from_labeled_tasks)

    print("initializing centroids from unconstrained data")
    # pick centers from data_unconstrained

    now = time.time()

    for i in range(n_clusters - num_centers_from_labeled_tasks):
        centers_array = np.array(cluster_centers)
        centers_array_rshp = centers_array.reshape((1, len(centers_array), dim))

        if i == 0:
            distances = pairwise_distances(data_unconstrained, centers_array)
            distances_squared = distances**2
            D = np.min(distances_squared, axis=1)
            j = sample_index_according_to_dist(D)
            new_cluster_center = data_unconstrained[j]
            cluster_centers.append(new_cluster_center)

        else:
            differences_to_new_center = data_unconstrained - new_cluster_center.reshape((1,-1))
            distances_squared_to_new_center = np.sum(differences_to_new_center**2, axis=1)

            double_D = np.stack([D, distances_squared_to_new_center], axis=0)

            D = np.min(double_D, axis=0)
            j = sample_index_according_to_dist(D)
            new_cluster_center = data_unconstrained[j]
            cluster_centers.append(new_cluster_center)

    print('initializing clusters time: {0:.0f} s'.format(time.time() - now))

    assert(len(cluster_centers) == n_clusters)

    centers_array = np.array(cluster_centers)

    print("beginning main loop")
    # ########################
    # main loop
    # ########################
    for i in range(max_iter):
        iter_time = time.time()
        print("iteration", i)
        ##########################
        # assign points to clusters
        # ########################
        repeat_assignment_step = True
        centers_were_reassigned = False
        while repeat_assignment_step:
            repeat_assignment_step = False
            cluster_assignments_constrained = []

            # assignment for constrained data:
            for b in range(num_tasks):
                current_task = data_constrained_concentrated[b]
                assert current_task.shape == (n_way, dim)

                distances = pairwise_distances(current_task, centers_array)
                max_distance = np.max(distances)
                for c in range(n_way):
                    current_row = distances[c]
                    chosen_j = np.argmin(current_row)
                    # make sure that same j is not chosen for any other c
                    distances[:, chosen_j] = max_distance + 1
                    cluster_assignments_constrained.extend([chosen_j] * num_samples_per_class)

            # assignment for unconstrained data:

            distances = pairwise_distances(data_unconstrained, centers_array)
            mean_distance = np.mean(distances)
            chosen_centers = np.argmin(distances, axis=1)
            cluster_assignments_unconstrained = chosen_centers.tolist()


            assignments_constrained_array = np.array(cluster_assignments_constrained, dtype=np.int32)
            assignments_unconstrained_array = np.array(cluster_assignments_unconstrained, dtype=np.int32)
            data_constrained_flat = data_constrained.reshape(-1,dim)

            counts = [np.count_nonzero(assignments_constrained_array == j) +
                      np.count_nonzero(assignments_unconstrained_array == j) for j in range(n_clusters)]
            sorted_indices = np.argsort(np.array(counts))

            number_empty_clusters = 0
            for j in range(n_clusters):
                if counts[sorted_indices[j]] == 0:
                    number_empty_clusters = number_empty_clusters + 1
                    repeat_assignment_step = True
                    new_center = centers_array[sorted_indices[n_clusters - 1 - j]] + np.random.randn(dim) * mean_distance / 1000.
                    centers_array[sorted_indices[j]] = new_center
                    centers_were_reassigned = True
            if repeat_assignment_step:
                print("i is", i)
                print("number empty clusters was ", number_empty_clusters)
                print("assigned new centers to empty clusters and repeating.")

        if i > 0:
            equal_constrained = assignments_constrained_array == assignments_constrained_array_old
            equal_unconstrained = assignments_unconstrained_array == assignments_unconstrained_array_old
            if equal_constrained.all() and equal_unconstrained.all() and not centers_were_reassigned:
                break

        assignments_constrained_array_old = assignments_constrained_array
        assignments_unconstrained_array_old = assignments_unconstrained_array

        # compute new cluster centers
        # all clusters are guaranteed to be nonempty due to the previous while loop
        data_constrained_flat = data_constrained.reshape((-1, dim))
        for j in range(n_clusters):
            new_center = (
                np.sum(data_constrained_flat[assignments_constrained_array == j], axis=0)
                +
                np.sum(data_unconstrained[assignments_unconstrained_array == j], axis=0)
            ) / float(counts[j])
            centers_array[j] = new_center

        # compute loss
        centers_assigned_to_unconstrained = [centers_array[assignments_unconstrained_array[j]]
                                             for j in range(assignments_unconstrained_array.shape[0])]
        centers_assigned_to_unconstrained = np.stack(centers_assigned_to_unconstrained)

        centers_assigned_to_constrained = [centers_array[assignments_constrained_array[j]]
                                           for j in range(assignments_constrained_array.shape[0])]
        centers_assigned_to_constrained = np.stack(centers_assigned_to_constrained)

        assert centers_assigned_to_unconstrained.shape == data_unconstrained.shape
        assert centers_assigned_to_constrained.shape == data_constrained_flat.shape

        diff_unconstrained = data_unconstrained - centers_assigned_to_unconstrained
        diff_constrained = data_constrained_flat - centers_assigned_to_constrained

        distances_unconstrained = np.linalg.norm(diff_unconstrained, axis=1)
        distances_constrained = np.linalg.norm(diff_constrained, axis=1)

        assert distances_constrained.shape == (num_tasks * n_way * num_samples_per_class,)
        assert distances_unconstrained.shape == (data_unconstrained.shape[0],)

        loss = np.sum(distances_constrained) + np.sum(distances_unconstrained)
        print('loss: {0:.3f}'.format(loss))
        print('iterations time: {0:.0f} s'.format(time.time() - iter_time))

    print('k-means time: {0:.0f} s'.format(time.time() - start_time))

    return assignments_constrained_array, assignments_unconstrained_array, loss

if __name__ == '__main__':

    data = np.load("embedding.npy")
    print("data shape is", data.shape)
    assert data.shape[0] == 64 * 600

    data_unconstrained = data[:52*600]

    for_labeled_use = data[52*600:]

    labeled_classes = [for_labeled_use[j*600:(j+1)*600] for j in range(12)]

    data_constrained = np.zeros((100, 5, 6, 256))

    for k in range(100):
        sampled_classes = random.sample(labeled_classes, 5)
        for j in range(5):
            data_constrained[k, j] = sampled_classes[j][k*6:(k+1)*6]

    cluster_with_constraints(data_constrained, data_unconstrained, 15, 500, 3000)

    










