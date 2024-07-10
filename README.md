# Logistic Regression Machine Learning

This repository contains a Python script for training a logistic regression model to predict social network ad clicks. The script includes steps for loading the dataset, splitting the data into training and test sets, feature scaling, and training the logistic regression model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Script Overview](#script-overview)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ShayanAlahyari/Logistic_Regression_Machine_Learning.git
    ```

2. Navigate to the repository directory:
    ```bash
    cd Logistic_Regression_Machine_Learning
    ```

3. Install the required dependencies:
    ```bash
    pip install numpy pandas matplotlib scikit-learn
    ```

## Usage

1. Place your dataset file (`Social_Network_Ads.csv`) in the repository directory.
2. Run the logistic regression script:
    ```bash
    python logistic_regression.py
    ```

## Script Overview

The `logistic_regression.py` script performs the following steps:

1. **Importing Libraries**:
    - Imports necessary libraries such as `numpy`, `pandas`, `matplotlib.pyplot`, and various `sklearn` modules.

2. **Creating the Dataset**:
    - Loads the dataset from `Social_Network_Ads.csv`.
    - Creates a matrix of features (`x`) and an output column (`y`).

3. **Splitting the Data**:
    - Splits the data into training and test sets using `train_test_split`.

4. **Feature Scaling**:
    - Scales the features using `StandardScaler`.

5. **Training the Model**:
    - Trains a logistic regression model on the training set using `LogisticRegression`.

## Dependencies

The script requires the following Python libraries:

- numpy
- pandas
- matplotlib
- scikit-learn

You can install these dependencies using `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn
