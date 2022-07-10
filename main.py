import knn
import dtree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import uniform

# CONSTANTS ===================
num_groups = 6
k = 15
dimensions = 5
num_points_in_cluster = 50
cluster_range = 20
ranges = (-50, 50)

# SETTINGS ====================
np.set_printoptions(precision=2)


def create_inputs(print_offsets: bool = True) -> np.ndarray:
    """ Creates a set of n-dimensional inputs grouped into k clusters"""
    X = np.zeros((num_points_in_cluster * num_groups, dimensions))

    if print_offsets:
        print("Expected: ")
    for i in range(0, num_groups):
        arr = cluster_range * np.random.rand(num_points_in_cluster, dimensions)
        randoms = np.array([uniform(*ranges) for _ in range(dimensions)])
        arr += randoms
        X[i * num_points_in_cluster:(i + 1) * num_points_in_cluster, :] = arr

        if print_offsets:
            print(randoms + cluster_range / 2)

    np.random.shuffle(X)
    return X


def create_and_train_model(model):
    model = model()
    X = create_inputs(True)
    model.fit(X, k, num_groups)

    test = model.create_random_cluster(model.ranges)
    if dimensions == 2:
        display_output(X, test, model)
    else:
        print("Outputs: ")
        model.print_results()

    return model


def display_output(inputs: np.ndarray, test_input: np.ndarray, model: knn.KNN) -> None:
    actual = model.centroids
    labels = model.labels

    sns.scatterplot(x=inputs[:, 0], y=inputs[:, 1], hue=labels)

    # plotting the centers
    for center in actual:
        plt.scatter(*center, c='r', s=10,)

    plt.scatter(*test_input, s=50, marker='o')
    print(model.classify(test_input, k))

    plt.show()


def main():
    # create_and_train_model(clustering.KMeansClustering)

    input_example = np.array(
        [
            [1, 1],
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0]
        ]
    )

    # 10 inputs with 2 choices, each with 2 options
    # print(dtree.DecisionTree.entropy(input_example))
    print(dtree.DecisionTree.information_gain(input_example, 1, 0))


if __name__ == '__main__':
    create_and_train_model(knn.KNN)
    # main()
