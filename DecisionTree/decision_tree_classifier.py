import numpy as np
from joblib import Parallel, delayed
from .node import Node

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=5, use_parallel=False):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.use_parallel = use_parallel
        self.root = None


    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''

        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:

            best_split = self.get_best_split(dataset, num_samples, num_features)

            if best_split["info_gain"] > 0:

                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)

                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)

                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])


        leaf_value = self.calculate_leaf_value(Y)

        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        ''' Function to find the best split, with optional parallelization '''
        if self.use_parallel:
            # Parallelize feature evaluation
            with Parallel(n_jobs=-1) as parallel:
                results = parallel(delayed(self.calculate_best_split_for_feature)(dataset, feature_index)
                                   for feature_index in range(num_features))
        else:

            results = [self.calculate_best_split_for_feature(dataset, feature_index)
                       for feature_index in range(num_features)]


        best_split = max(results, key=lambda x: x["info_gain"], default=None)
        return best_split

    def calculate_best_split_for_feature(self, dataset, feature_index):
        ''' function to calculate best split for a given feature '''

        feature_values = dataset[:, feature_index]
        possible_thresholds = np.unique(feature_values)
        best_split_for_feature = {"info_gain": -float("inf")}

        for threshold in possible_thresholds:
            dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
            if len(dataset_left) > 0 and len(dataset_right) > 0:
                y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                if curr_info_gain > best_split_for_feature["info_gain"]:
                    best_split_for_feature = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "dataset_left": dataset_left,
                        "dataset_right": dataset_right,
                        "info_gain": curr_info_gain
                    }

        return best_split_for_feature

    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def entropy(self, y):
        ''' function to compute entropy '''
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def gini_index(self, y):
        ''' function to compute gini index '''

        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls ** 2
        return 1 - gini

    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        ''' function to train the tree '''
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        ''' function to predict new dataset '''
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
