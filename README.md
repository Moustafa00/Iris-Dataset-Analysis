# Iris Dataset Analysis

## Overview

This project aims to predict the class of iris flowers based on their characteristics such as sepal length, sepal width, petal length, and petal width. We will analyze the dataset, visualize the data, and build several machine learning models to determine the most accurate model for classification.

## Problem Statement

Predict the class of flower (Iris-setosa, Iris-versicolor, or Iris-virginica) based on its features: sepal length, sepal width, petal length, and petal width.

## Libraries Used

- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For plotting and visualization.
- **Scikit-learn**: For machine learning models and evaluation metrics.

## Dataset

The dataset used is the Iris dataset, which can be accessed from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data). It consists of 150 samples of iris flowers with 4 features each.

## Steps

1. **Load Libraries**

    ```python
    import pandas as pd
    from pandas.plotting import scatter_matrix
    import matplotlib.pyplot as plt
    from sklearn import model_selection
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    ```

2. **Load Dataset**

    ```python
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    ```

3. **Explore Dataset**

    - Shape of the dataset
    - Display first 30 rows
    - Class distribution
    - Univariate plots: Box and whisker plots, Histograms
    - Multivariate plots: Scatter plot matrix

4. **Create Validation Dataset**

    ```python
    array = dataset.values
    X = array[:,0:4]
    Y = array[:,4]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    ```

5. **Build and Evaluate Models**

    - Models used: Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Decision Tree Classifier, Gaussian Naive Bayes, Support Vector Machine
    - Evaluate models using 10-fold cross-validation
    - Compare algorithms using box plots

6. **Make Predictions**

    - Train the K-Nearest Neighbors model
    - Predict on the validation dataset
    - Evaluate accuracy, confusion matrix, and classification report

7. **Conclusion**

    The K-Nearest Neighbors model achieved an accuracy of 90% on the validation dataset. The confusion matrix and classification report provide detailed performance metrics, showing that the model performs well across different classes.

## Results

- **Accuracy**: 90%
- **Confusion Matrix**:

    ```plaintext
    [[ 7  0  0]
     [ 0 11  1]
     [ 0  2  9]]
    ```

- **Classification Report**:

    ```plaintext
                  precision    recall  f1-score   support
      Iris-setosa       1.00      1.00      1.00         7
  Iris-versicolor       0.85      0.92      0.88        12
   Iris-virginica       0.90      0.82      0.86        11
      avg / total       0.90      0.90      0.90        30
    ```


## References

- [Iris Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)


## Portfolio

Curious about what I do? Dive into my portfolio [here](https://moustafa00.github.io/) and stay up-to-date.


