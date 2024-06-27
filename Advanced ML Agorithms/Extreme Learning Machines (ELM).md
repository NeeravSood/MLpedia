# Extreme Learning Machines (ELM)

**Algorithm Name**: Extreme Learning Machines (ELM)  
**Type**: Supervised Learning  
**Category**: Classification/Regression

## Overview

Extreme Learning Machines (ELM) offer a fast and efficient single-hidden layer feedforward neural network. Historically developed by Guang-Bin Huang and colleagues in the early 2000s, ELMs are known for their rapid training times and excellent generalization performance. They are primarily used to address regression and classification problems, particularly where training speed is a critical factor.

## Key Concepts

- **Random Hidden Nodes**: ELM uses randomly generated weights and biases in the hidden nodes, which are not iteratively adjusted.
- **Analytical Weight Determination**: The output weights are analytically determined through a direct calculation rather than traditional iterative approaches like backpropagation.
- **Generalization Capability**: Despite their simplicity, ELMs can achieve comparable or superior generalization performance to traditional feedforward networks with significantly reduced training times.

## How It Works

1. **Random Initialization**: Randomly assign input weights and biases to hidden nodes.
2. **Hidden Layer Transformation**: Transform the input features into the hidden layer space using nonlinear activation functions.
3. **Linear System Solution**: Use a least squares method to analytically determine the output weights that map the hidden layer outputs to the target outputs.

## Mathematical Model (Optional)

Given a training set \((x_i, t_i)\), the output function of ELM for binary class classification can be modeled as:

\[ o_i = \sum_{j=1}^L \beta_j h_j(x_i) \]

where \( h_j(x) \) is the output of the \( j \)-th hidden neuron and \( \beta_j \) are the output weights computed to minimize the difference between the actual output and the target.

## Pseudocode

```plaintext
BEGIN
    Randomly initialize input weights and biases
    FOR each input x_i
        Calculate hidden layer outputs h_i(x)
    END FOR
    Calculate output weights β using least squares method
RETURN model
END
```

## Implementation (Language-Specific)

### Python Example

```python
import numpy as np

def elm_train(X, T, num_hidden_units):
    # Randomly generate weights and biases
    input_weights = np.random.rand(num_hidden_units, X.shape[1])
    biases = np.random.rand(num_hidden_units)
    H = np.tanh(np.dot(X, input_weights.T) + biases)
    # Calculate output weights (β)
    output_weights = np.dot(np.linalg.pinv(H), T)
    return output_weights

def elm_predict(X, input_weights, biases, output_weights):
    H = np.tanh(np.dot(X, input_weights.T) + biases)
    return np.dot(H, output_weights)
```

## Applications

- **Pattern Recognition**: ELMs are widely used in biometric systems and image classification due to their quick training and good performance.
- **Regression**: Useful in predicting continuous variables in various industrial and economic sectors due to fast computation.
- **Feature Learning**: ELMs can be applied in unsupervised feature learning scenarios to enhance learning in deep networks.

## Strengths

- **Speed**: Significantly faster training compared to traditional neural networks due to the elimination of iterative weight adjustment.
- **Simplicity**: Easy to implement and requires minimal hyperparameter tuning.

## Limitations

- **Randomness**: The random nature of hidden node parameters can lead to variability in performance.
- **Scalability**: Although fast, large-scale problems might suffer from computational inefficiencies in solving linear systems.

## References and Further Reading

- Huang, G.-B., et al. (2006): "Extreme learning machine: a new learning scheme of feedforward neural networks."
- Zhang, R., et al. (2015): "Theory and applications of extreme learning machine: a survey."
```
