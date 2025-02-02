# CPU-Based Decision Tree Classifier

## Features
✅ Implements a **Decision Tree Classifier** using NumPy and OpenMP.
✅ Supports **multi-threading** for parallel execution.
✅ Uses **MNIST dataset** for handwritten digit classification.
✅ Performance measurement tools for **execution time and resource utilization**.
✅ Optimized for **AMD Ryzen CPUs**, but adaptable to other architectures.

## Installation Prerequisites
- **Python 3.10+**
- **NumPy (2.2.2)**
- **Pandas (2.2.3)**
- **scikit-learn (1.6.1)**
- **TensorFlow (2.18.0)**
- **OpenMP (MinGW GCC 6.3.0)** *(for parallel execution)*


## Usage
To execute the CPU-based Decision Tree Classifier, run:
```bash
python main.py
```

### Performance Testing
The script will output execution times and accuracy metrics:
```bash
Non-parallelized execution time: 210.99 seconds
Optimized execution time: 62.17 seconds
Accuracy: 0.6400
```

## File Structure
```
├── DecisionTree
│   ├── decision_tree_classifier.py
│   ├── node.py
├── main.py
├── README.md
```

## Benchmarks
| Metric | Non-Optimized CPU | Optimized CPU |
|--------|------------------|--------------|
| Execution Time | ~210 sec | ~62 sec |
| Speedup Ratio | 3.4x | - |

## Future Improvements
- Further optimize **OpenMP parallelization** for better CPU efficiency.

## Contributors
- **Klejda Rrapaj** - k.rrapaj@student.unisi.it
- **Sildi Riçku** - s.ricku@student.unisi.it
