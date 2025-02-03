import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from DecisionTree.decision_tree_classifier import DecisionTreeClassifier
import time
from tensorflow.keras.datasets import mnist

# 1. Data Preparation
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

subset_size = 1000
X_train, X_test, Y_train, Y_test = train_test_split(
    x_train[:subset_size], y_train[:subset_size], test_size=0.2, random_state=41
)

pca = PCA(n_components=15)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# 2. Model Training and Evaluation
classifier_no_parallel = DecisionTreeClassifier(min_samples_split=3, max_depth=5)

start_time = time.time()
classifier_no_parallel.fit(X_train, Y_train)
end_time = time.time()
non_parallel_time = end_time - start_time

Y_pred_no_parallel = classifier_no_parallel.predict(X_test)
non_parallel_accuracy = accuracy_score(Y_test, Y_pred_no_parallel)

classifier_parallel = DecisionTreeClassifier(min_samples_split=3, max_depth=5, use_parallel=True)

start_time = time.time()
classifier_parallel.fit(X_train, Y_train)
end_time = time.time()
parallel_time = end_time - start_time

Y_pred_parallel = classifier_parallel.predict(X_test)
parallel_accuracy = accuracy_score(Y_test, Y_pred_parallel)

# 3. Metrics Definition
metrics = {
    "Non-Parallel Execution Time (seconds)": non_parallel_time,
    "Non-Parallel Accuracy": non_parallel_accuracy,
    "Parallel Execution Time (seconds)": parallel_time,
    "Parallel Accuracy": parallel_accuracy,
}

for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")