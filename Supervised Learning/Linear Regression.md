# Linear Regression

**Type**: Supervised Learning  
**Category**: Regression

## Overview
Linear Regression is a statistical method that models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. The origins of linear regression date back to the 19th century, primarily developed by Francis Galton.

## Key Concepts
- **Least Squares**: The method minimizes the sum of the squares of the differences between observed and predicted values.
- **Regression Coefficients**: Parameters estimated from the data that quantify the impact of independent variables.
- **Goodness of Fit**: Measures how well the regression model represents the data (e.g., R-squared).

## How It Works
1. **Data Collection**: Gather data points (dependent and independent variables).
2. **Model Fitting**: Use least squares to fit a line through the data.
3. **Interpretation**: Analyze the slope and intercept of the line to interpret the effects of variables.

## Mathematical Model
```
y = β0 + β1x1 + ε
```
Where `y` is the dependent variable, `x1` is the independent variable, `β0` is the intercept, `β1` is the slope, and `ε` is the error term.

## Pseudocode
```
function linear_regression(data):
    calculate beta coefficients using least squares
    return model
```

## Implementation (Python Example)
```python
import numpy as np

def linear_regression(X, y):
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta
```

## Applications
- **Economics**: Predicting consumer spending.
- **Healthcare**: Estimating medical expenses based on lifestyle.
- **Real Estate**: Pricing homes based on location and features.

## Strengths
- Simplicity: Easy to implement and interpret.
- Efficiency: Requires relatively simple calculations.

## Limitations
- Linearity Assumption: Only models linear relationships.
- Sensitive to Outliers: Outliers can significantly impact the regression line.

## References and Further Reading
- "An Introduction to Statistical Learning"
- Original Paper on Linear Regression by Francis Galton
