import math
import sys

def run(k, vectors):
    vectors = vectors.tolist()
    iter = 300
    eps = 0.0001
    N = len(vectors)
    try:
        # initialization
        iteration_number = 0
        mu = float("inf")
        centroids = vectors[:k]
        if k > len(vectors):
            sys.exit("An Error Has Occurred")
        # running the k-means algorithm while mu > eps or number of iterations < iter
        while iteration_number < iter or mu > eps:
            clusters = [[] for i in range(k)]
            mu, centroids = compute_k_means(clusters, centroids, vectors)
            iteration_number += 1

    except:
        sys.exit("An Error Has Occurred")

    return centroids


def find_cluster(vec, centroids):
    """
    :param vec: a vector
    :param centroids: a list of centroids
    :return: the index of the closest centroid
    """
    min_dist = float("inf")
    min_dist_idx = 0
    for cen in centroids:
        dist = math.sqrt(sum([math.pow((vec[i]-cen[i]), 2) for i in range(len(vec))]))
        if dist < min_dist:
            min_dist = dist
            min_dist_idx = centroids.index(cen)
    return min_dist_idx

def compute_new_centroid(cluster):
    """
    :param cluster: a cluster
    :return: the new cluster's centroid
    """
    new_centroid = []
    for i in range(len(cluster[0])):
        sum_cord = 0
        for vec in cluster:
            sum_cord += vec[i]
        new_centroid.append(sum_cord / len(cluster))
    return tuple(new_centroid)


def compute_mu(old_centroid, new_centroid):
    """
    :param old_centroid: the old centroid
    :param new_centroid: the new centroid
    :return: the mu value, as defined in the exercise
    """
    return math.sqrt(sum([math.pow((old_centroid[i] - new_centroid[i]), 2) for i in range(len(old_centroid))]))

def compute_k_means(clusters, centroids, vectors):
    """
    :param clusters: a list of clusters
    :param centroids: a list of centroids
    :param vectors: a list of the input vectors
    :return: mu - the maximal mu value of all centroids
             new_centroids - a list of the centroids after a k-means iteration
    """
    for i in range(len(vectors)):  # for each vector
        clus = find_cluster(vectors[i], centroids)  # find cluster
        clusters[clus].append(vectors[i])  # adding the current vector to it's cluster

    max_mu = float("-inf")
    new_centroids = []
    for i in range(len(clusters)):
        new_centroids.append(compute_new_centroid(clusters[i]))
        mu = compute_mu(centroids[i], new_centroids[i])
        if mu > max_mu:
            max_mu = mu

    return mu, new_centroids

