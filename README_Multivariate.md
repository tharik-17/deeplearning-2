# Multivariate Linear Regression: House Price Prediction

A from-scratch implementation of multivariate linear regression with batch gradient descent in NumPy. The model predicts the price of a house from two features: its size in square feet and its number of bedrooms.

This is the follow-up to the single-variable [food truck profit notebook](README.md) and is based on the second part of Exercise 1 in Andrew Ng's Coursera Machine Learning course.

## Overview

The notebook (`Multivariate_Linear_Regression.ipynb`) walks through:

1. Loading the dataset
2. Feature normalization (z-score scaling)
3. Preparing the design matrix `X` and target vector `y`
4. Implementing the cost function and batch gradient descent
5. Training the model and checking convergence with a cost-vs-iterations plot

## Dataset

`data/ex1data2.txt` is a comma-separated file with no header:

| Column   | Description                |
|----------|----------------------------|
| Size     | House size in square feet  |
| Bedrooms | Number of bedrooms         |
| Price    | Sale price of the house    |

## Feature Normalization

House size is in the thousands while the number of bedrooms is a single digit. Without scaling, the larger feature dominates the gradient updates and gradient descent converges slowly. Every column is standardized by subtracting its mean and dividing by its standard deviation:

```python
df = (df - df.mean()) / df.std()
```

Because the target `Price` is scaled in the same step, the reported cost is in normalized units, not dollars. To predict on new data, apply the same means and standard deviations to the inputs, then reverse the scaling on the predicted price.

## Method

**Hypothesis:** `h(x) = theta0 + theta1 * size + theta2 * bedrooms`

**Cost function:**

```
J(theta) = (1 / 2m) * sum((X * theta^T - y)^2)
```

**Gradient descent update (applied to every parameter simultaneously):**

```
theta_j := theta_j - (alpha / m) * sum((X * theta^T - y) * x_j)
```

A column of ones is prepended to `X` so the intercept is handled by the matrix product.

## Hyperparameters

| Parameter                | Value       |
|--------------------------|-------------|
| Learning rate (`alpha`)  | 0.01        |
| Iterations               | 1000        |
| Initial theta            | `[0, 0, 0]` |

## Results

| Metric                         | Value   |
|--------------------------------|---------|
| Final cost (normalized scale)  | ~0.1307 |

The cost curve drops steeply in the early iterations and then flattens, which confirms that gradient descent is converging.

## Requirements

- Python 3.5+
- NumPy
- pandas
- Matplotlib
- Jupyter Notebook

```bash
pip install numpy pandas matplotlib notebook
```

## Usage

1. Place the dataset at `data/ex1data2.txt`.
2. Launch Jupyter:

   ```bash
   jupyter notebook Multivariate_Linear_Regression.ipynb
   ```

3. Run all cells in order.

> **Note:** The notebook reads the data with a Windows-style path (`data\ex1data2.txt`). On macOS or Linux, change it to `data/ex1data2.txt`, or use `os.path.join('data', 'ex1data2.txt')`.

## Project Structure

```
.
├── Multivariate_Linear_Regression.ipynb
├── data/
│   └── ex1data2.txt
└── README_Multivariate.md
```

## Possible Improvements

- Pass `X` and `y` into the cost calculation inside `gradientDescent`. It currently uses the global `X1` and `y1`, so the function only works for that one dataset.
- Print the learned `theta` values and show how to predict a price for a new house.
- Try several learning rates (for example 0.003, 0.01, 0.03, 0.1) and compare their convergence curves.
- Compare the result with the closed-form normal equation, which needs no feature scaling.
- Replace `np.matrix` with `np.ndarray`, since `np.matrix` is deprecated.

## Reference

- Andrew Ng, *Machine Learning*, Coursera (Exercise 1: Linear Regression, multiple variables)
