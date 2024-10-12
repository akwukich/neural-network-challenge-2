# neural-network-challenge-2

## Background

The purpose of this challenge was to create a branched neural network that HR can use to predict (1) whether employees are likely to leave the company and (2) the department that best fits each employee.

## Installation

The following libraries and dependencies are required to successfully run the project:
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import StandardScaler
- import pandas as pd
- import numpy as np
- from tensorflow.keras.models import Model
- from tensorflow.keras import layers



## Repository Files and Starter Code
- [Module 19 Starter Code](https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/starter/M19_Starter_Code.zip)
- attrition.ipynb : completed workbook

## Methodology
The project was completed in three parts:
### 1. Preprocessing
- Import the data and read it into the "attrition_df" DataFrame.
- Determine the number of unique values in each column to determine the kind of classification and complexity.
- Created y_df with the "Attrition" and "Department" columns. In order to avoid issues later with the model, I set the index to the original "attrition_df" DataFrame.
- Created X_df using your 10 columns from the original DataFrame: 'Age', 'BusinessTravel', 'DistanceFromHome', 'Education', 'HourlyRate', 'JobLevel', 'JobSatisfaction', 'PerformanceRating', 'WorkLifeBalance', 'YearsAtCompany'.
- I explored the X_df Show the data types for X_df. I used the .info function to get a summary of X data to ensure there were no issues with null values and the data types for each column. The only non-numerical column was "BusinessTravel." I also noted that the X_df contained 1470 rows of data.
- I set my X, or features, to the X_df DataFrame. For this assignment, we were concerned with two output variables, or targets, "Department" and "Attrition," so I created two y variables: y_dept and y_att. I ensured that both y variables were set to the same index as the X_df so the rows aligned correctly between the features and targets. I then split the data into testing and training sets: 80% of the data was used for training, and 20% was used for testing. During testing and training, I maintained the distinction between both targets (y_dept and y_att) so they would continue to be treated independently.
- The only feature that needed to be converted to numerical values was "BusinessTravel." Even though the three classes 'Travel_Rarely', 'Travel_Frequently', 'Non-Travel' could be considered ordinal, I was concerned that if I used LabelEncoder, I would have to map the classes to ensure the correct order was preserved. This seemed like extra and unnecessary effort since this was just one of ten features, and not a target, and therefore I elected to use OneHotEncoder. I convert your X data to numeric data types however you see fit. Add new code cells as necessary. Make sure to fit any encoders to the training data, and then transform both the training and testing data.
- I then scaled the data by creating an instance of StandardScaler, fit the scaler to the training data, and then transform both the training and testing data.
- Then, according to the instructions, I created a OneHotEncoder for the both the "Department" and "Attrition" columns, then fit the encoder to the respective training data and used the to transform the training and testing data for both columns.


### 2. Create, Compile, Train, and evaluate a model using neural network
To create the layers for my model I:
- Found the number of columns in the X training data, which was 12. I created the input layer and two shared layers.
- Created a branch to predict the Department target column with one hidden layer and one output layer. For the Department output I used the "softmax" activation function because it was a multiclass.
- Created a branch to predict the Attrition target column with one hidden layer and one output layer. For the Attrition output I used the "softmax" activation function because it was a multiclass.
Next, I:
- Created the model using these inputs and output layers.
- Compiled the model using the Adam optimizer and separate loss functions for each target. I used Categorical Crossentropy for the multiclass Department target, and Binary Crossentropy for the binary Attrition target. I used the accurcy metric to evalue the model and its performance.
- Summarized the model.
- Trained the model using the preprocessed data.
- Evaluated the model with the testing data.



## Evaluation and Discussion

**1. Is accuracy the best metric to use on this data? Why or why not?**

Accuracy is probably not the best metric to use here, particularly for the Department target. Department is multi-class, and the data is unbalanced with the majority falling into Research & Development. Accuracy is a better metric for Attrition because it is a simpler, binary classification of Yes or No. Even this classification might warrant a different metric because ethe Attrition data is so unbalanced (1233 No vs. 237 Yes).

**2. What activation functions did you choose for your output layers, and why?**

For the Department output layer, I used the softmax activation function because this target was multi-classification, and the output would be one class rather than a mixed or hybrid classification. For the Activation output layer, I used sigmoid because it was a binary classification target.

**3. Can you name a few ways that this model might be improved?**

There are a few things I would try in order to improve the model which has high loss and low accuracy (not saying they will all work, but would be good places to start).

Since the model contains such unbalanced data we could:
- Try using class weights for the Department Category or resampling to minimize the impact of the unbalanced data.
- Add training data, change the percentage of training/testing data, or reduce the number of eophs.
- Improve the training data by identifying alternative features, or perhaps scaling the current features.

Use another metric instead of accuracy to better understand the model's performance and where potential issues lie.

Fine tune the model: use hyperparameters (i.e. Keras tuner) to optimize the number of neurons, the number of hidden layers, the appropriate activation function, and the number of epochs.

## Resources Consulted
For this challenge, I consulted
- [Module 19 Starter Code](https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/starter/M19_Starter_Code.zip) and my in-class slides, notes, and activities from Modules 14, 18 and 19.
