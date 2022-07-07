import numpy as np
from random import uniform
import warnings


class KMeansClustering:

    def __init__(self):
        self.centroids = np.array([])
        self.ranges = []

    def fit(self, inputs: np.ndarray, k: int, max_iterations: int = 250) -> None:
        """ Takes in an array of inputs and clusters it into k clusters"""

        self.ranges = self.input_range(inputs)
        self.create_initial_clusters(k, self.ranges)

        for _ in range(max_iterations):
            # making a copy of previous clusters
            previous_clusters = np.copy(self.centroids)

            self.iterate(inputs)

            if self.reached_threshhold(previous_clusters):
                break
        else:
            warnings.warn("Did not converge. Consider increasing maximum iterations.")

    def print_results(self) -> None:
        list(map(print, self.centroids))


    def iterate(self, inputs: np.ndarray):
        """ Iterates the centroids """
        labels = self.label_distribution(inputs)
        self.update_centroids(inputs, labels)

    @staticmethod
    def create_random_cluster(ranges: list[tuple[float, ...]]) -> tuple[float, ...]:
        return tuple(uniform(*r) for r in ranges)

    def create_initial_clusters(self, k: int, ranges: list[tuple[float, ...]]) -> None:
        """ Sets the centroid clusters instance variable"""
        self.centroids = [self.create_random_cluster(ranges) for _ in range(k)]

    def label_distribution(self, distribution: np.ndarray) -> np.ndarray:
        """ Labels each point in the distribution according to the cluster it is closest to"""
        return np.array([self.closest_cluster_index(p, self.centroids) for p in distribution])

    def update_centroids(self, distribution: np.ndarray, labels: np.ndarray) -> None:
        """ Moves each centroid to the average of each cluster"""
        grouped_clusters = {x: [] for x in range(len(self.centroids))}
        list(map(lambda x, y: grouped_clusters[x].append(y), labels, distribution))

        cluster_averages = {x: self.average_of_points(grouped_clusters[x]) for x in range(len(self.centroids))}

        for k, v in cluster_averages.items():
            self.centroids[k] = v

    def reached_threshhold(self, previous_clusters: np.ndarray, tolerance: float = 0.1) -> bool:
        """ Returns whether the clustering has converged"""
        distances = np.abs(previous_clusters - self.centroids)
        little_moved = distances < tolerance
        return np.all(little_moved)

    @staticmethod
    def input_range(inputs: np.ndarray) -> list[tuple[float, float]]:
        """ Returns the minimum and maximum of each dimension"""

        def update_range(n_min: float, n_max: float, num: float) -> tuple[float, float]:
            """ Updates the range of a single dimension"""
            return min(n_min, num), max(n_max, num)

        def update_range_multi_d(multi_d_ranges: list[tuple[float, float]],
                                 nums: np.ndarray) -> list[tuple[float, float]]:
            """ Updates the range of multiple dimension input"""
            return [update_range(mini, maxi, n) for (mini, maxi), n in zip(multi_d_ranges, nums)]

        num_dimensions = len(inputs[0])
        ranges = [(0, 0)] * num_dimensions

        for points in inputs:
            ranges = update_range_multi_d(ranges, points)
        return ranges

    def average_of_points(self, p_list: np.ndarray) -> np.ndarray:
        """ Returns the mean of all the points"""
        if len(p_list) == 0:
            return np.array(list(self.create_random_cluster(self.ranges)))
        return np.divide(sum(p_list), len(p_list))

    @staticmethod
    def closest_cluster_index(p: np.ndarray, clusters: np.ndarray):
        """ Returns the index (int) of the closest cluster to the specified point"""
        return np.argmin([KMeansClustering.euclidian_distance_squared(p, q) for q in clusters])

    @staticmethod
    def euclidian_distance_squared(p: np.ndarray, q: np.ndarray) -> float:
        """ Returns the distance squared between two points"""
        return sum(map(lambda a, b: (a - b) ** 2, p, q))
