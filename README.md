# GMM Clustering Project

This project is the course assignment for the Pattern Recognition course at Tongji University. It involves implementing the Gaussian Mixture Model (GMM) clustering algorithm on various datasets from the UCI Machine Learning Repository.

**Student Information:**
- **University**: Tongji University, Software Engineering

## Table of Contents

- [Usage](#usage)
- [Data Sets](#data-sets)
- [Features](#features)

## Usage

1. Clone the repository or download the project files.

2. Make sure to install the dependencies. To run this project, you'll need to install the required dependencies. You can do this via `pip`:

```bash
pip install -r requirements.txt
```

3. The main file to run is `main.py`. You can execute it in your terminal or command prompt:

```bash
python main.py
```

## Selecting Different Datasets

In the `main.py` file, you can modify the `dataset_id` in the following line to select different datasets from the UCI Machine Learning Repository:

```python
X, y, metadata = load_dataset(dataset_id=53)
```

To select a different dataset, change the `dataset_id` to the desired dataset's ID (e.g., `dataset_id=45` for a different dataset).

## Example Output

Running the program will print the following to the terminal:

```yaml
The dataset has 3 unique targets: ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
Number of clusters determined by target classes: 3
Converged at iteration 26
Adjusted Rand Index (ARI): 0.9038742317748124
```
Clustering results have been visualized.

It will also show clustering visualizations in individual windows (one for each pair of features if the dataset has more than two features).

## Data Sets

The project uses datasets from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml). Currently, two example datasets are used:

1. **Iris Dataset** (ID: 53)  
   A classic dataset for classification and clustering tasks, containing 150 samples of iris flowers, with 4 features and 3 target classes.

2. **Breast Cancer Wisconsin (Diagnostic)** (ID: 17)  
   A dataset containing 569 instances of wine, with 30 features and 2 target classes.
    
2. **Wine Dataset** (ID: 109)  
   A dataset containing 178 instances of wine, with 13 features and 3 target classes.
    
You can modify the `load_dataset()` function to load other datasets from UCI by specifying the dataset's ID.