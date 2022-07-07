import clustering
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from typing import Type, Protocol, ClassVar

# CONSTANTS ===================
k = 6
dimensions = 5
num_points_in_cluster = 50
cluster_range = 5
ranges = (-50, 50)

# SETTINGS ====================
np.set_printoptions(precision=2)


def create_inputs(print_offsets: bool = True) -> np.ndarray:
    """ Creates a set of n-dimensional inputs grouped into k clusters"""
    X = np.zeros((num_points_in_cluster * k, dimensions))

    if print_offsets:
        print("Expected: ")
    for i in range(0, k):
        arr = cluster_range * np.random.rand(num_points_in_cluster, dimensions)
        randoms = np.array([uniform(*ranges) for _ in range(dimensions)])
        arr += randoms
        X[i * num_points_in_cluster:(i + 1) * num_points_in_cluster, :] = arr

        if print_offsets:
            print(randoms + cluster_range / 2)

    np.random.shuffle(X)
    return X


class Model(Protocol):
    centroids: ClassVar[np.ndarray]

    def fit(self, inputs: np.ndarray, *args, **kwargs) -> None:
        ...

    def print_results(self) -> None:
        ...


def create_and_train_model(model: Type[Model]) -> Model:
    model = model()
    X = create_inputs()
    model.fit(X, k)

    print("Outputs: ")
    model.print_results()

    if dimensions == 2:
        display_output(X, model.centroids)

    return model


def display_output(inputs: np.ndarray, actual: np.ndarray) -> None:
    plt.scatter(inputs[:, 0], inputs[:, 1])

    # plotting the centers
    for center in actual:
        plt.scatter(*center, c='r', s=10)

    plt.show()


def main():
    create_and_train_model(clustering.KMeansClustering)


if __name__ == '__main__':
    main()
