# Sailing Through Time: A Titanic Data Odyssey

---

## Project Description

**Description:** Our project was centered around the analysis of the Titanic dataset, which we obtained from Kaggle.com. Our primary research objective was to investigate the determinants of passenger survival and explore the application of machine learning techniques for predictive modeling. Specifically, we aimed to train a logistic regression model to discern patterns in the data and predict the likelihood of passenger survival, thus contributing to a deeper understanding of the factors influencing survival outcomes.

---

## Directory

---

## Setup:
- Imported the data from Kaggle Dataset - Titanic Machine Learning from Disaster
- Conducted exploratory data analysis to gain insight into the different classifications that were provided in the dataset
- Used tableau to visualize age and gender distribution, age group and gender by socio economic class, fare distribution, survival by gender and non survival by both socio economic class and gender.
- Used the logistic regression model to predict which passengers will survive and which will not.
- We used the random over sampler to resample the data.
- Tested the predictive power of socioeconomic status and the predictive power of gender.

---

## Dependencies

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report

## Input Files

- Training set (train.csv) - dataset representing outcome for each passenger.

---

## Tableau Visualizations and Jupyter Notebook Files
- titanic data - tableau visualizations
- titanic_survival_classification.ipynb
- titanic_survival_classification_no_pclass.ipynb
- titanic_survival_classification_no_null.ipynb
  
  ---

## Exploratory Analysis Charts
- 

---

## Analysis Presentation file

- [PPT](./Project%204/Powerpoint.pptx)
  
---

## Data Prep and Analysis Steps

- Loaded the CSV file  located on kaggle.com
- Split the Data into Training and Testing Sets¶
- Step 1: Read the data from the Resources folder into a Pandas DataFrame
- Step 2: Create the labels set (y) from the “Survival” column, and then create the features (X) DataFrame from the remaining columns
- Step 3: Check the balance of the labels variable (y) by using the value_counts function
- Step 4: Split the data into training and testing datasets by using train_test_split
- Create a Logistic Regression Model with the Original Data
- Step 1: Fit a logistic regression model by using the training data (X_train and y_train)
- Step 2: Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model
- Step 3: Evaluate the model’s performance by doing the following
- Predict a Logistic Regression Model with Resampled Training Data
- Step 1: Use the RandomOverSampler module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points
- Step 2: Use the LogisticRegression classifier and the resampled data to fit the model and make predictions
- Step 3: Evaluate the model’s performance by doing the following

  ## Insights

  **Age group and gender distribution**
  









