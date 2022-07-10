from clustering import KMeansClustering
import numpy as np


class KNN(KMeansClustering):

    def __init__(self):
        super().__init__()
        self.labels: list = []
        self.inputs: np.ndarray = ...

    def fit(self, inputs: np.ndarray, k: int, n: int = 0, max_iterations: int = 250) -> None:
        self.inputs = inputs
        if n == 0:
            n = k

        super().fit(inputs, n)
        self.labels = self.label_distribution(inputs)

    def classify(self, x: np.ndarray, k: int):

        distances = [(i, KMeansClustering.euclidian_distance_squared(x, q)) for i, q in enumerate(self.inputs)]
        distances.sort(key=lambda d: d[1])
        labels = [self.labels[a[0]] for a in distances[:k]]
        return max(set(labels), key=labels.count)

