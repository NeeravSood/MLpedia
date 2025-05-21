### **Algorithm Name:** Random Forests

**Type:** Supervised Learning
**Category:** Classification/Regression

---

### **Overview**

Random Forests are an ensemble learning method that operates by constructing a multitude of decision trees during training and outputting the mode of the classifications (for classification tasks) or mean prediction (for regression tasks) of the individual trees. Introduced by Leo Breiman in 2001, Random Forests aim to reduce overfitting and increase predictive accuracy by averaging multiple trees built on randomly sampled data.

---

### **Key Concepts**

* **Ensemble Learning:** Combines predictions from multiple models to improve accuracy.
* **Bootstrap Aggregation (Bagging):** Each tree is trained on a random subset of the training data.
* **Feature Randomness:** At each split in a tree, only a random subset of features is considered, promoting tree diversity.
* **Voting/Averaging:** Predictions are combined using majority vote (classification) or mean (regression).

---

### **How It Works**

1. Randomly sample (with replacement) data points from the training set to build multiple subsets (bootstrap samples).
2. Train a decision tree on each bootstrap sample.
3. At each node split, select a random subset of features to determine the best split.
4. Aggregate predictions from all trees:

   * **Classification:** Use majority voting.
   * **Regression:** Use the mean of outputs.

---

### **Mathematical Model**

Let $T_1(x), T_2(x), ..., T_n(x)$ be the predictions from $n$ individual decision trees.

* **Classification Output:**
  $\hat{y} = \text{mode} \{ T_1(x), ..., T_n(x) \}$
* **Regression Output:**
  $\hat{y} = \frac{1}{n} \sum_{i=1}^n T_i(x)$

---

### **Pseudocode**

```
for i in 1 to n_trees:
    sample data with replacement (bootstrap)
    grow decision tree with feature randomness
aggregate predictions from all trees
```

---

### **Implementation (Language-Specific)**

Example code snippet in Python.

**Python Example:**

```python
def random_forest_example(features, target):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, target)
    return model
```

---

### **Applications**

* Credit scoring and risk assessment
* Medical diagnostics
* Stock market prediction
* Customer segmentation
* Text classification and spam detection

---

### **Strengths**

* **High Accuracy:** Reduces overfitting through ensemble averaging.
* **Robust to Noise:** Handles missing values and irrelevant features well.
* **Scalable:** Parallelizable and efficient for large datasets.

---

### **Limitations**

* **Less Interpretable:** Difficult to interpret compared to a single decision tree.
* **Computational Cost:** Training multiple trees can be time-consuming and memory-intensive.
* **Biased with Imbalanced Data:** May favor dominant classes if not properly handled.

---

### **References and Further Reading**

**Leo Breiman’s Original Paper on Random Forests:** \[[Link](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)]
