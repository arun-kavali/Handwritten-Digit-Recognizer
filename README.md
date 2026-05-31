# 🔢 Digit Recognizer using Machine Learning | Kaggle Competition

## 📌 Overview

This project implements a Handwritten Digit Recognition System using Machine Learning on the popular MNIST dataset provided through Kaggle's Digit Recognizer competition.

The objective is to accurately classify handwritten digits (0–9) based on pixel values from grayscale images. The project covers the complete machine learning workflow including data preprocessing, model training, hyperparameter tuning, prediction generation, and Kaggle competition submission.

---

## 🎯 Project Objective

Develop a machine learning model capable of recognizing handwritten digits with high accuracy by learning patterns from labeled image data.

The trained model predicts unseen handwritten digits and generates submission files compatible with Kaggle's Digit Recognizer competition.

---

## 📊 Dataset

Dataset Source:

**Kaggle Digit Recognizer Competition**

The dataset is based on the MNIST handwritten digit dataset.

### Training Dataset

Contains:

* 42,000 labeled images
* Each image represented by 784 pixel values
* Labels range from 0 to 9

### Test Dataset

Contains:

* 28,000 unlabeled images
* Used for prediction and Kaggle evaluation

### Image Characteristics

* Grayscale images
* Resolution: 28 × 28 pixels
* Flattened into 784 numerical features

---

## ⚙️ Machine Learning Workflow

### 1. Data Collection

Loaded training and testing datasets using Pandas.

```python
train_df = pd.read_csv('train.csv')
test_X_df = pd.read_csv('test.csv')
```

---

### 2. Data Preparation

Separated:

* Features (Pixel Values)
* Target Labels (Digits)

```python
train_X_df = train_df.drop('label', axis=1)
train_Y_df = train_df['label']
```

---

### 3. Train-Validation Split

Used Scikit-Learn's train_test_split() to create validation data for model evaluation.

```python
from sklearn.model_selection import train_test_split

train_X_split, validation_X_split, train_Y_split, validation_Y_split = train_test_split(
    train_X_df,
    train_Y_df,
    test_size=0.2
)
```

---

### 4. Model Training

Implemented the K-Nearest Neighbors (KNN) Classification Algorithm.

```python
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(
    n_neighbors=1,
    p=2
)
```

---

### 5. Hyperparameter Tuning

Used GridSearchCV to identify the best-performing parameters.

Parameters Tested:

```python
{
    'n_neighbors': [10, 15],
    'p': [1, 2]
}
```

Best Parameters Found:

```python
{
    'n_neighbors': 10,
    'p': 2
}
```

Grid Search enabled improved prediction accuracy through systematic parameter optimization.

---

### 6. Model Prediction

Generated predictions for Kaggle test data using the optimized KNN model.

```python
predicted_test_Y = best_model.predict(test_X_df)
```

---

### 7. Submission File Generation

Created a Kaggle-compatible submission file.

```python
submission.csv
```

Format:

| ImageId | Label |
| ------- | ----- |
| 1       | 7     |
| 2       | 2     |
| 3       | 1     |

---

### 8. Individual Image Prediction

Implemented prediction functionality for individual handwritten digit images.

Features:

* Image visualization
* Digit prediction
* Grayscale rendering using Matplotlib

---

## 🧠 Machine Learning Algorithm

### K-Nearest Neighbors (KNN)

KNN classifies a digit by examining the closest training examples in the feature space.

Key Parameters:

| Parameter       | Value              |
| --------------- | ------------------ |
| n_neighbors     | 10                 |
| Distance Metric | Euclidean Distance |
| p               | 2                  |

Advantages:

* Simple and effective
* No training assumptions
* High accuracy on image classification tasks

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Scikit-Learn

### Machine Learning

* K-Nearest Neighbors (KNN)
* GridSearchCV
* Train-Test Split

### Platform

* Kaggle

### Development Environment

* Google Colab
* Jupyter Notebook

---

## 📈 Project Features

✅ Handwritten Digit Classification

✅ MNIST Dataset Processing

✅ Data Splitting and Validation

✅ Hyperparameter Optimization

✅ Automated Prediction Generation

✅ Kaggle Submission File Creation

✅ Individual Digit Prediction

✅ Image Visualization

---

## 🚀 How to Run

### Clone Repository

```bash
git clone <repository-url>
cd digit-recognizer
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Download Dataset

Download:

* train.csv
* test.csv

From the Kaggle Digit Recognizer Competition.

Place both files inside the project directory.

### Run Notebook

```bash
jupyter notebook
```

or

```bash
python digit_recognizer.py
```

---

## 📊 Results

* Successfully trained a handwritten digit classification model.
* Optimized model performance using Grid Search.
* Generated Kaggle-compatible prediction files.
* Achieved accurate digit recognition on unseen test images.
* Demonstrated end-to-end machine learning workflow from training to deployment-ready prediction generation.

---

## 🔮 Future Improvements

* Implement Convolutional Neural Networks (CNN)
* Improve classification accuracy using Deep Learning
* Deploy model as a web application
* Real-time handwritten digit recognition
* Mobile integration for digit scanning
* TensorFlow and PyTorch implementations

---

## 👨‍💻 Author

**Arun Kavali**

Machine Learning | Data Science | Artificial Intelligence

---

## 🏆 Learning Outcomes

This project provided hands-on experience with:

* Supervised Machine Learning
* Image Classification
* Feature-Based Learning
* Hyperparameter Tuning
* Model Evaluation
* Kaggle Competitions
* Data Preprocessing
* Scikit-Learn Workflow

It demonstrates the complete implementation of a machine learning solution for handwritten digit recognition using the MNIST dataset.
