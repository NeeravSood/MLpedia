Type: Supervised Learning
Category: Regression Models

Overview
Support Vector Regression (SVR) is a type of Support Vector Machine (SVM) that supports regression tasks. This method involves predicting a continuous variable, unlike the traditional SVM that is used for classification tasks.

Key Concepts
- **Support Vectors**: Data points that are closer to the hyperplane and influence the position and orientation of the hyperplane.
- **Hyperplane**: In SVR, this is the decision surface where the objective is to fit the largest possible number of data points.
- **Kernel Trick**: Used to solve nonlinear problems by transforming them into a higher dimensional space to make it possible to perform linear separation.
- **Error Margin / Epsilon**: A threshold within which errors are tolerated in SVR. The model tries to fit the data within this margin.

How It Works
SVR works by trying to fit the error within a certain threshold (epsilon). The model will create a hyperplane in such a way that the errors of the prediction from the actual values are minimized and contained within this margin. It uses kernel functions to handle non-linear relationships.

Mathematical Model
Pseudocode
```
initialize SVR model with parameters C (penalty), kernel type, epsilon
for training iterations do
    select support vectors that define the hyperplane
    minimize error within the defined margin
update model based on support vectors and margin constraints
```

Implementation
```python
from sklearn.svm import SVR
import numpy as np

# Sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

# Fit regression model
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
y_rbf = svr_rbf.fit(X, y).predict(X)
```

Applications
- Predicting stock prices
- Estimating house values
- Forecasting sales and other business metrics

Strengths
- Effective in high-dimensional spaces.
- Works well with non-linear data due to the kernel trick.
- Robust against overfitting, especially in high-dimensional space.

Limitations
- Requires careful selection of parameters.
- Not suitable for large datasets as the required computation time is higher.
- Sensitive to the type of kernel used.

References and Further Reading
- Drucker, H., Burges, C.J.C., Kaufman, L., Smola, A., and Vapnik, V., "Support Vector Regression Machines," Advances in Neural Information Processing Systems, 1997.
- Smola, A. J., and Schölkopf, B., "A tutorial on support vector regression," Statistics and computing, 2004.
