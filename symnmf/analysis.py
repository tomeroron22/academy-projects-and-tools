import numpy as np
import symnmf
import kmeans
import sys
from sklearn.metrics import silhouette_score


def symnmf_clusters_func(x, k):
	"""
    :param x: a ndarray of the input vectors
    :type x: ndarray
    :param k: number of clusters
    :type k: int
    The function returns a dictionary - keys: clusters indices, values: vectros, using the symnmf method
    """
	H = symnmf.symnmf(x.tolist(), k)
	clusters = {}
	H = np.array(H)
	return H.argmax(axis=1)


def kmeans_clusters_func(k, x):
	"""
    :param x: a ndarray of the input vectors
    :type x: ndarray
    :param k: number of clusters
    :type k: int
    The function returns a dictionary - keys: clusters indices, values: vectros, using the kmeans method
    """
	centroids = kmeans.run(k, x)
	clusters = {}
	flags = [kmeans.find_cluster(vector, centroids) for vector in x]
	return flags


def main(argv):
	"""
    Main function to perform anlysis and compare between kmeans and symnmf as required.
    :param argv: Command-line arguments containing [script_name, k, file_path].
    :type argv: list
    """
	file_path = argv[2]
	k = int(argv[1])
	x = np.loadtxt(file_path, delimiter=",")
	if not (2 <= k <= len(x) - 1):
		sys.exit("An Error Has Occurred")
	symnmf_clusters = symnmf_clusters_func(x, k)
	kmeans_clusters = kmeans_clusters_func(k, x)
	N = len(x)

	print(f"nmf: {np.around(silhouette_score(x, symnmf_clusters), 4)}")
	print(f"kmeans: {np.around(silhouette_score(x, kmeans_clusters), 4)}")


if __name__ == "__main__":
	main(sys.argv)