import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from DecisionTree.decision_tree_classifier import DecisionTreeClassifier
import time
import socket
import subprocess
import psutil
from tensorflow.keras.datasets import mnist
import os

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

# System Check Functions
def check_system_resources():
    print("Checking system resources before experiment...")
    if os.name == "posix":  # Linux/MacOS
        print("Logged-in users:")
        subprocess.run(["w"], text=True)
        print("\nSystem resource usage:")
        subprocess.run(["top", "-b", "-n", "1"], text=True)
        print("\nMemory status:")
        subprocess.run(["free", "-h"], text=True)
        print("\nDisk usage:")
        subprocess.run(["df", "-h"], text=True)
        print("\nProcess limits:")
        subprocess.run(["ulimit", "-a"], text=True)
    elif os.name == "nt":  # Windows
        print("\nSystem Information:")
        subprocess.run(["systeminfo"], text=True)
        print("\nTask List (Running Processes):")
        subprocess.run(["tasklist"], text=True)
        print("\nMemory Usage:")
        subprocess.run(["wmic", "OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize"], text=True)
        print("\nDisk Usage:")
        subprocess.run(["wmic", "logicaldisk", "get", "size,freespace,caption"], text=True)

def generate_experiment_tag():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    hostname = socket.gethostname()
    return f"EXP-{timestamp}-{hostname}"

def log_resource_usage(stage):
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk_usage = psutil.disk_usage('/')
    log_data = f"""
    Experiment Stage: {stage}
    CPU Usage: {cpu_usage}%
    Memory Usage: {memory_info.percent}%
    Disk Usage: {disk_usage.percent}%
    """
    with open("resource_usage_log.txt", "a") as log_file:
        log_file.write(log_data + "\n")

# Experiment Preparation
check_system_resources()
experiment_id = generate_experiment_tag()
print(f"Running experiment with ID: {experiment_id}")
with open("experiment_log.txt", "a") as log_file:
    log_file.write(f"{experiment_id}, CPU: AMD Ryzen 5, Dataset: MNIST\n")
log_resource_usage("Before Training")

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

log_resource_usage("After Training")

# 3. Metrics Definition
metrics = {
    "Non-Parallel Execution Time (seconds)": non_parallel_time,
    "Non-Parallel Accuracy": non_parallel_accuracy,
    "Parallel Execution Time (seconds)": parallel_time,
    "Parallel Accuracy": parallel_accuracy,
}

for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")
