import numpy as np
from typing import ClassVar
from itertools import product


class DecisionTree:
    """ A classification algorithm meant to be used only on discreet variables"""

    def __init__(self):
        # creates a dictionary with the key being the (choice, index) and another dictionary of choices
        self.choices: ClassVar[dict[tuple[int, int], dict[...]]] = {}

    def fit(self, inputs: np.ndarray, labels: np.ndarray, max_iterations: int = 250):
        """ Takes in an array of inputs and clusters it into k clusters"""

        while True:
            if len(inputs) == 1:
                break

            choice, index = self.find_best_split(inputs)
            chosen, not_chosen = self.choose(inputs, choice, index)
            self.choices.append((choice, index))

    def find_best_split(self, inputs: np.ndarray) -> tuple[int, int]:
        """ Returns the choice and the index of that choice that maximizes information gain"""
        num_choices, num_dimensions = inputs.size()
        all_choices = list(product(range(num_choices + 1), range(num_dimensions + 1)))

        entropies = list(map(lambda choice: self.information_gain(inputs, *choice), all_choices))
        best = np.argmin(entropies)

        return all_choices[best]

    def evaluate(self, inputs: np.ndarray) -> int:
        """ Returns the category of inputs"""

        if not self.choices:
            raise ValueError("Did not fit before trying to evaluate outputs")

        for choice in self.choices:
            inputs = self.choose(inputs, *choice)
        print(inputs)
        raise NotImplementedError

    def printResults(self):
        pass

    @staticmethod
    def entropy(inputs: np.ndarray) -> float:
        """ Returns the amount of entropy remaining """

        def entropy_single(x: float | np.ndarray) -> float:
            """ Returns the entry along a single probability"""
            return (x * np.log2(x))[0] if x else 0

        def num_choices(x: np.ndarray) -> int:
            return np.max(x) + 1

        def number_each_choice(x: np.ndarray, axis: int):
            """ Returns a list of counts for each choice"""
            return [np.sum(x[:, axis].reshape((-1,)) == i) for i in range(int(num_choices(x)))]

        num_inputs, num_dimensions = inputs.shape
        nums = np.divide([number_each_choice(inputs, d) for d in range(num_dimensions)], num_inputs).reshape((-1, 1))

        return -1 * sum([entropy_single(y) for y in nums])

    @staticmethod
    def choose(inputs: np.ndarray, choice: int, index: int) -> tuple[np.ndarray, np.ndarray]:
        """ Returns the chosen inputs (and the not chosen ones)"""
        num_inputs = len(inputs)
        mask = np.array(inputs[:, index].reshape((num_inputs,)) == choice)
        return inputs[mask], inputs[~mask]

    @staticmethod
    def information_gain(inputs: np.ndarray, choice: int, index: int) -> float:
        """ Returns the amount of information gained from the inputs"""
        start = DecisionTree.entropy(inputs)
        end = DecisionTree.entropy(DecisionTree.choose(inputs, choice, index)[0])

        # higher number is better
        return start - end
