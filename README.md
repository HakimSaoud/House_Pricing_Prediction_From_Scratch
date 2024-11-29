# Multiple Linear Regression Project

## Project Overview

This project implements a simple **Multiple Linear Regression** model from scratch in Python using **NumPy** and **Pandas**. The model is trained on housing data (`houses.csv`) to predict a target variable (e.g., house prices) based on selected features (e.g., size, number of rooms, etc.).

The project demonstrates how to:
1. Load and preprocess data using `Pandas`.
2. Split data into training and testing sets.
3. Build and train a linear regression model using gradient descent.
4. Evaluate the model by predicting on test data.

---

## Files in the Repository

- **`houses.csv`**: The dataset used for training and testing the model.
- **`linear_regression.py`**: The Python script containing the implementation of the linear regression model.
- **`README.md`**: Documentation for the project.

---

## How It Works

### 1. Data Preparation
The dataset (`houses.csv`) is loaded using `pandas.read_csv`. The target variable `y` is extracted as the first column, while the features `X` are selected as the last two columns. The dataset is split into training (80%) and testing (20%) sets.

### 2. Model Implementation
The **`MultipleLinearRegression`** class includes:
- **Initialization (`__init__`)**: Sets the learning rate and initializes weights and bias.
- **Forward Pass (`Z`)**: Calculates predictions as the dot product of input features and weights, plus bias.
- **Cost Function (`cost_function`)**: Computes the Mean Squared Error (MSE).
- **Gradient Calculation (`gradient`)**: Derives gradients for weights and bias using partial derivatives of MSE.
- **Training (`fit`)**: Updates weights and bias iteratively using gradient descent.
- **Prediction (`predict`)**: Predicts the target values for new input data.

### 3. Training and Prediction
The model is trained on the training set using `fit`. The predictions on the test set are then rounded to simplify the output.

---

## Requirements

To run the project, you need:
- **Python 3.7+**
- **NumPy**
- **Pandas**

Install the dependencies using:

```bash
pip install numpy pandas
```
## Usage

Clone the repository:

```bash
git clone https://github.com/your-username/linear-regression.git
cd linear-regression
```
- **Place your dataset in the same directory and name it houses.csv.**

## Run the script:

```bash
python linear_regression.py
```
View the model predictions printed to the console.

## Key Functions and Methods
- **```fit(X_train, y_train, epochs)```: Trains the model using the provided training data and the specified number of epochs.**
- **```predict(X_test)```: Returns predictions for the test set.**
## Notes
- Ensure the dataset ```houses.csv``` is correctly formatted with numerical columns, where the first column is the target variable.
- Adjust the learning rate ```lr``` and epochs to fine-tune model performance.

## License
-This project is open-source under the MIT License.
-Feel free to contribute or suggest improvements! 😊

