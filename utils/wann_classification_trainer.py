from copy import copy
import json
import numpy as np
from os import mkdir, listdir
from os.path import exists, join
from random import uniform
import tensorflow as tf
from typing import List, Tuple


from wann import WANN


class WANNClassificationTrainer:
    """
    Training population of WANNs to solve classification task
    """

    def __init__(self, sensor_nodes: List[str],
                 output_nodes: List[str],
                 num_iterations: int = 1000,
                 batchsize: int = 1024,
                 weights: Tuple[float] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0),
                 checkpoint_dir: str = './checkpoint',
                 start_from_checkpoint: bool = False,
                 visualize_history: bool = False):
        """
        :param sensor_nodes: sensor nodes names to use for WANN creation
        :param output_nodes: output nodes names to use for WANN creation
        :param num_iterations: number of iterations to spend on WANNs training
        :param batchsize: minibatch size of data to use during training
        :param weights: WANNs shared weights
        :param checkpoint_dir: directory, where
        :param start_from_checkpoint:
        :param visualize_history:
        """
        self.sensor_nodes = sensor_nodes
        self.output_nodes = output_nodes
        self.num_wanns = 10
        self.num_iterations = num_iterations
        self.batchsize = batchsize
        self.weights = weights
        self.checkpoint_dir = checkpoint_dir
        self.start_from_checkpoint = start_from_checkpoint
        self.visualize_history = visualize_history

        self.wanns = None
        self.history = None

    def _save_checkpoint(self) -> None:
        """
        Save current state of training in a checkpoint directory
        """

        # Creating directory if it doesn't exist
        if not exists(self.checkpoint_dir):
            mkdir(self.checkpoint_dir)

        # Saving current history
        with open(join(self.checkpoint_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)

        # Saving WANNs
        for i, wann in enumerate(self.wanns):
            with open(join(self.checkpoint_dir, f"WANN_{i}.json"), 'w') as f:
                json.dump(wann.to_dict(), f, indent=4)

    def _load_checkpoint(self) -> None:
        """
        Load training state from checkpoint directory
        """
        assert exists(self.checkpoint_dir), f"Directory '{self.checkpoint_dir}' doesn't exist"

        # Loading history
        with open(join(self.checkpoint_dir, 'history.json')) as f:
            self.history = json.load(f)

        # Loading WANNs
        self.wanns = []
        for filename in listdir(self.checkpoint_dir):
            with open(join(self.checkpoint_dir, filename)) as f:
                self.wanns.append(WANN(input_dict=json.load(f)))

    def _init_random_wanns(self) -> None:
        """
        Initialize new random WANNs before training
        """
        # assert self.wanns is None or len(self.wanns) == 0, f"There are already existing WANNs in the list"
        self.wanns = []
        for _ in range(self.num_wanns):
            wann = WANN(self.sensor_nodes, self.output_nodes, output_activation='softmax').init_randomly()
            self.wanns.append(wann)

    def fit(self, X, y) -> List[WANN]:
        """
        Fit WANNs population to solve given classification task
        :param X: data to train on
        :param y: correct labels for given data (expecting not onehot-encoded)
        :return: list of WANNs, acquired after last iteration
        """

        # WANNs initialization, either from a checkpoint of from scratch
        if self.start_from_checkpoint:
            self._load_checkpoint()
        else:
            self._init_random_wanns()

        y_onehot = np.zeros((len(y), 10), dtype=np.float32)
        y_onehot[:, y] = 1.0

        # History object, that keep losses and accuracies at all iterations
        self.history = dict(loss=[], accuracy=[], complexity=[])

        # Training
        print(f"Training {self.num_wanns} WANNs for {self.num_iterations} iterations".center(100, '-'))
        for iteration in range(self.num_iterations):

            loss = [0.] * self.num_wanns
            accuracy = [0.] * self.num_wanns

            self._save_checkpoint()

            for i, wann in enumerate(self.wanns):

                wann.build_tf_graph()

                # Batch walking
                for idx in range(0, len(X), self.batchsize):

                    _X = X[idx:idx + self.batchsize]
                    _y = y[idx:idx + self.batchsize]
                    _y_onehot = y_onehot[idx:idx + self.batchsize]

                    for weight in self.weights:
                        results = wann.run(_X, weight)
                        loss[i] += np.sum(np.linalg.norm(_y_onehot - results, axis=1))
                        accuracy[i] += np.sum(_y == np.argmax(results, axis=1))

                loss[i] /= len(X) * len(self.weights)
                accuracy[i] /= len(X) * len(self.weights)

            # Adding current iteration results to history
            complexity = [wann.complexity for wann in self.wanns]
            iter_loss = np.mean(loss)
            iter_accuracy = np.mean(accuracy)
            iter_complexity = int(round(np.mean(complexity)))
            self.history['loss'].append(iter_loss)
            self.history['accuracy'].append(iter_accuracy)
            self.history['complexity'].append(iter_complexity)
            print(f"Iteration {iteration}:"
                  f" loss = {iter_loss},"
                  f" accuracy = {iter_accuracy},"
                  f" complexity = {iter_complexity}")

            # Creating list of evaluated WANNs and sorting it by mean accuracy
            evaluated_wanns = sorted(list(zip(self.wanns, loss, accuracy, complexity)),
                                     key=lambda p: p[2], reverse=True)

            # Cutting WANNs with low accuracy (7 WANNs are left)
            del evaluated_wanns[-2:]

            # Choosing evaluation method
            evaluation_method = "accuracy" if 0 <= uniform(0, 1) < 0.8 else "complexity"

            # Evaluating remaining WANNs according to evaluation method
            if evaluation_method == "complexity":

                # Sorting WANNs by increasing complexity
                evaluated_wanns.sort(key=lambda p: p[3])

            elif evaluation_method == "accuracy":

                # Sorting WANNs by decreasing accuracy
                evaluated_wanns.sort(reverse=True, key=lambda p: p[1])

            # Cutting WANNs with small maximum performance of with big complexity
            del evaluated_wanns[-2:]

            tf.reset_default_graph()

            # Creating new WANNs by mutating and filling new list with them
            new_wanns = []
            for i in range(len(evaluated_wanns)):

                # Creating mutations while its not unique
                mutated_wann = copy(evaluated_wanns[i][0]).mutate()
                while mutated_wann in new_wanns or mutated_wann in self.wanns:
                    mutated_wann = copy(evaluated_wanns[i][0]).mutate()
                new_wanns.append(mutated_wann)

                if i < 4:  # First WANNs have right to mutate twice   ODD CHECKING MIGHT BE ADD

                    # Creating mutations while its not unique
                    mutated_wann = copy(evaluated_wanns[i][0]).mutate()
                    while mutated_wann in new_wanns or mutated_wann in self.wanns:
                        mutated_wann = copy(evaluated_wanns[i][0]).mutate()
                    new_wanns.append(mutated_wann)

            # Deleting evaluated WANNs
            for ewann, _, _, _ in evaluated_wanns:
                ewann.clear_tf_graph()
            del evaluated_wanns

            # Deleting all self.wanns
            del self.wanns
            self.wanns = new_wanns

        return self.wanns
