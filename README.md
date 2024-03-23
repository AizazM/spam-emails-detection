# Spam Email Detection with Classification Models

This repository implements and evaluates various machine learning classification models for spam email detection using Python.

## Classification Models

The following classification models are utilized for spam email detection:

1. **Logistic Regression Classification**: A binary classification algorithm that models data using the sigmoid function, suitable for predicting whether an email is spam or non-spam based on its features.

2. **Decision Tree Classification**: Decision trees are employed to split email data based on certain parameters, aiding in classifying emails as spam or non-spam. They recursively partition the data into subsets, making decisions based on attribute values.

3. **Random Forest Classification**: Random Forest is an ensemble learning technique that combines multiple decision trees to enhance spam detection accuracy. It builds multiple decision trees and combines their predictions to provide a more accurate and stable result.

4. **K-nearest Neighbor (KNN) Classification**: KNN algorithm is used to determine whether an email is spam by considering the classification of its nearest neighbors. It calculates the distance between the input email and all other emails in the dataset and assigns it to the majority class among its k-nearest neighbors.

5. **Support Vector Machine (SVM) Classification**: SVM constructs hyperplanes to separate spam and non-spam emails in high-dimensional space. It maximizes the margin between the classes to find the optimal decision boundary, making it effective for both linear and nonlinear classification tasks.

6. **Naive Bayes Classification**: Naive Bayes algorithm provides a probabilistic approach to predict spam emails based on the features of the email content. It assumes that the presence of a particular feature in an email is independent of the presence of other features, making it computationally efficient and easy to implement.

## Model Evaluation

The performance of each classification model is evaluated using the following techniques:

- **Classification Accuracy**: Measures the fraction of correctly classified spam and non-spam emails.
- **Confusion Matrix**: Provides a detailed summary of correct and incorrect predictions made by each classification model.
- **Classification Report**: Offers precision, recall, F1-score, and support scores, providing insights into the performance of each model.
- **Receiver Operating Characteristics (ROC) Curve**: Visualizes the trade-off between true positive rate and false positive rate for different classification thresholds, with ROC AUC indicating overall model performance.

## References

For detailed explanations and implementations of each classification model and evaluation technique, refer to the following resources:

- [GeeksforGeeks - Logistic Regression](https://www.geeksforgeeks.org/understanding-logistic-regression/)
- [Decision Tree - Medium](https://medium.com/swlh/decision-tree-classification-de64fc4d5aac)
- [Random Forest Algorithm - BuiltIn](https://builtin.com/data-science/random-forest-algorithm)
- [KNN Classification - DataCamp](https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn)
- [SVM Classification - DataCamp](https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python)
- [Naive Bayes - DataCamp](https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn)
- [Model Evaluation Techniques - Towards Data Science](https://towardsdatascience.com/model-evaluation-techniques-for-classification-models-eac30092c38b)

