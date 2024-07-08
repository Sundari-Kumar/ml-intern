# Machine Learning Projects: House Price Prediction & Iris Classification

This repository contains two separate projects: one for predicting house prices and another for classifying Iris species. Both projects demonstrate different machine learning techniques and provide insights into data preprocessing, model training, and evaluation.

## Table of Contents
- [Introduction](#introduction)
- [House Price Prediction](#house-price-prediction)
  - [Data Visualization](#data-visualization)
  - [Preprocessing](#preprocessing)
  - [Machine Learning Models](#machine-learning-models)
    - [Batch Gradient Descent](#batch-gradient-descent)
    - [Stochastic Gradient Descent](#stochastic-gradient-descent)
    - [Mini Batch Gradient Descent](#mini-batch-gradient-descent)
  - [Neural Network Model](#neural-network-model)
- [Iris Classification](#iris-classification)
  - [Data Exploration](#data-exploration)
  - [Preprocessing](#preprocessing-1)
  - [Machine Learning Models](#machine-learning-models-1)
    - [Logistic Regression](#logistic-regression)
    - [Decision Tree](#decision-tree)
    - [Random Forest](#random-forest)
- [Usage](#usage)
- [Requirements](#requirements)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction

This repository contains two distinct projects:
1. **House Price Prediction**: Predicting house prices based on features such as area and number of bedrooms.
2. **Iris Classification**: Classifying Iris species based on sepal and petal dimensions.

## House Price Prediction

### Data Visualization

Initial data visualization includes:
- Plotting area vs. price while keeping the number of bedrooms constant.
- Plotting the number of bedrooms vs. price by bucketing the area into intervals.

### Preprocessing

The preprocessing step includes normalizing the features to ensure they have a mean of 0 and a standard deviation of 1. This is crucial for the convergence of gradient descent algorithms and training neural networks.

### Machine Learning Models

#### Batch Gradient Descent

Implemented in the function `batch_gradient_descent`, this algorithm updates the weights by computing the gradients of the loss function for the entire dataset.

#### Stochastic Gradient Descent

Implemented in the function `sgd_one_sample`, this algorithm updates the weights using a single sample at each iteration, which helps in faster convergence for large datasets.

#### Mini Batch Gradient Descent

Implemented in the function `mini_batch_gd`, this algorithm updates the weights using small batches of the dataset, combining the advantages of both batch and stochastic gradient descent.

### Neural Network Model

A neural network model is built using TensorFlow and Keras, consisting of:
- An input layer.
- Two hidden layers with ReLU activation.
- An output layer for predicting the house price.

The model is trained using the Adam optimizer and Mean Squared Error loss function.

## Iris Classification

### Data Exploration

Data exploration includes:
- Visualizing the distribution of features for different Iris species.
- Analyzing the pairwise relationships between features.

### Preprocessing

Preprocessing steps include standardizing the features to improve model performance and handling any missing values if present.

### Machine Learning Models

#### Logistic Regression

A simple yet powerful algorithm for binary and multiclass classification problems.

#### Decision Tree

A non-linear model that splits the data into subsets based on feature values to make predictions.

#### Random Forest

An ensemble model that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

## Usage

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/machine-learning-projects.git
    cd machine-learning-projects
    ```

2. Ensure you have the required data files in the correct directories:
    - `bharath intern/ex1data2.txt` for the house price prediction project.
    - `iris.csv` for the Iris classification project.

3. Install the necessary packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Run the house price prediction model:
    ```sh
    python house_price_prediction.py
    ```

5. Run the Iris classification model:
    ```sh
    python iris_classification.py
    ```

## Requirements

- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow (for neural network in house price prediction)
- seaborn (for Iris classification visualizations)

Install the required packages using:
```sh
pip install -r requirements.txt
