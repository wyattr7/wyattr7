
*Note: this notebook can be exectuted cell by cell, or all at once by selecting Runtime -> Run All. Some cell may be hidden; they can be unhidden by clicking on the cell.*
"""

#connect to shared drive

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

# load data
df = pd.read_csv("/content/drive/Shareddrives/The Mail Order Miracle/Data/case9_data_updated.csv")
df

"""## **Exploratory Analysis**"""

# summary statistics
df.describe()

"""Examine missing values."""

df.isnull().sum()

"""## **Data cleaning**"""

web_use_percent_null = df["WebUse"].isnull().sum()/len(df["WebUse"])
print("Percent of null values in WebUse:", web_use_percent_null*100,"%")

"""Because > 98% of the values are null, the column WebUse is being dropped."""

df.drop(["WebUse"], axis = 1, inplace = True)

"""The columns Age, LastMail, monthfrstord, and monthlastord are imputed for completing missing values using k-nearest neighbors.

*Note: the KNNImputer algorithim is relatively slow, this cell may take a while to execute.*
"""

from sklearn.impute import KNNImputer

cols_to_imputate = ['Age', 'LastMail', 'monthfrstord', 'monthlastord']
impute = df[cols_to_imputate]

for col in impute:

  imputer = KNNImputer(n_neighbors=2)
  arr = np.array(impute[col])
  arr = arr.reshape(-1,1)
  arr = imputer.fit_transform(arr)
  df[col] = arr

# post-impuatation data
df[cols_to_imputate]

"""**Because the imputation may take a while, we can avoid repeating that process by saving the dataframe post-imputation.**"""

import os
os.chdir("/content/drive/Shareddrives/Data/The Mail Order Miracle")

file_name = "case9_data_post_impute.csv"
df.to_csv(file_name, index=False)

## if you want to avoid repeating the imputation process, uncomment the lines below and load the post-imputation df ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/content/drive/Shareddrives/The Mail Order Miracle/Data/case9_data_post_impute.csv")

"""Next, we need to deal with the ProdCat columns. It is useful to first plot these columns to understand the data."""

df["ProdCatB"].plot()

df["ProdCatC"].plot()

df["ProdCatD"].plot()

df["ProdCatE"].plot()

"""For the ProdCat columns, we will use a binning technique."""

def make_bins(col):

  binned = []

  for value in col:

    if value == np.nan:       # if value is null, return 0
      value = 0
      binned.append(value)
    if value <= 100:          # if value <= 100, return 1
      value = 1
      binned.append(value)
    else:                     # for all remaining values, return 2
      value = 2
      binned.append(value)

  return binned

# make bins
df["ProdCatB"] = make_bins(df["ProdCatB"])
df["ProdCatC"] = make_bins(df["ProdCatC"])
df["ProdCatD"] = make_bins(df["ProdCatD"])
df["ProdCatE"] = make_bins(df["ProdCatE"])

df.isnull().sum()

"""Finally, we will drop the remaining 3 rows that contain a missing region value."""

df.dropna(axis = 0, how = 'any', inplace = True)

"""We now have no missing values."""

df.isnull().sum()

"""Next we need to investigate the data types of our columns."""

df.dtypes

"""Most of our columns have a numeric data type. Those that are not must be represented numerically to be useable in machine learning algorithms."""

non_numeric_cols = ["AgeCode", "gender", "homeval", 
                    "incgrp", "maritalgrp", "NoLonger",
                    "region", "state", "UseNet"]
df[non_numeric_cols]

"""Take a look at variable levels."""

for col in df[non_numeric_cols]:
  print(col)
  print("Levels:", df[col].unique())
  print("")

"""Some variables contain lots of levels, so we will use label encoding to convert the variables."""

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

for col in non_numeric_cols:
  df[col] = label_encoder.fit_transform(df[col])

"""These columns are now represented numerically."""

for col in df[non_numeric_cols]:
  print(col)
  print("Levels:", df[col].unique())
  print("")

"""## **Model Building**

We are using a neural network architecture built with Tensorflow and Keras. The justification for using a neural network is that these algorithms generally perform well on classification problems and can uncover patterns and associations in the data that traditional machine learning algorithms are unable to. This can lead to better predictive power.

Before we get started, it is useful to again take a look at our summary statistics.
"""

df.describe()

"""After looking at the summary statistics, it is clear that we should take the log of ZLabel to reduce its range. To do so, we will make a copy of the dataframe and make that change to make sure we are not tampering with the original data."""

# copy df
cleaned_df = df.copy()
cleaned_df.drop("ZLabel", axis=1, inplace=True)
# take log of ZLabel
#cleaned_df['ZLab'] = np.log(cleaned_df.pop('ZLabel'))

"""Next, we will take a look at how many positive samples (ie Resp = 1) exist in the data."""

neg, pos = np.bincount(df['Resp'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

"""Next, we split the dataframe into training and testing sets and define our features (x) and label(s) (y)."""

from sklearn.model_selection import train_test_split

# 75% of the data is used for training the model and the remaining 25% is used for testing the model
# we are stratifying the split based on the Resp column to ensure the same proportion of positive samples in both the training and testing sets
# random state is set for reproducability
train_df, test_df = train_test_split(cleaned_df, test_size=0.2, stratify=cleaned_df["Resp"], random_state=1)

# seperate labels and features and convert to arrays
train_labels = np.array(train_df["Resp"])
test_labels = np.array(test_df["Resp"])

train_features = np.array(train_df.loc[ : , train_df.columns != "Resp"])
test_features = np.array(test_df.loc[ : , test_df.columns != "Resp"])

# check shapes
print("Train features shape:", train_features.shape)
print("Train label(s) shape:", train_labels.shape)
print("Test features shape:", test_features.shape)
print("Test label(s) shape:", test_labels.shape)

"""Since some of our features have different ranges, we will normalize all of our features which will set the mean to 0 and standard deviation to 1. This will help to make sure the model is not placing too much emphasis on one feature based on that feature having much larger values than the others."""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Test features shape:', test_features.shape)

"""Import necessary libraries for model building."""

import tensorflow as tf
import keras
from keras import initializers

# set random seed for reproducability
tf.random.set_seed(1)

"""Here we define the architecture of the model."""

# define metrics
METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.AUC(name='auc')
]

# model architecture
def make_model(metrics=METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  
  model = keras.Sequential([
      keras.layers.Dense(1024, activation='relu',input_shape=(train_features.shape[-1],)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
  ])

  # compile model
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-4),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  return model

"""We saw above that only 3.56% of the samples in the training data are positive. This means we have a class imbalance. To account for this, we will apply different weights to the positive and negative samples. We will also intentionally bias the model to make it aware of the fact that the vast majority of the samples are negative."""

weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}
class_weight

# calculate bias based on the number of pos and neg samples
initial_bias = np.log([pos/neg])
initial_bias

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta = 0.0001, patience=20, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.00001)

"""We can now fit the model on the training data.

*Note: the model takes a few min to train*
"""

# define batch size
BATCH_SIZE = 2048
EPOCHS = 200

# call make_model function
model = make_model(output_bias=initial_bias)

# model summary shows us all of the model components
model.summary()

# fit model with training data
history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight)

"""We can now plot the results of the model training."""

def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_result("loss")
plot_result("auc")
plot_result('accuracy')

"""Next, the model can be evaluated on the test dataset and predicted probabilities can be generated."""

model.evaluate(test_features, test_labels)

# generate predicted probabilites for samples in the test set
predictions = model.predict(test_features)

# any predicted probability > 0.5 indicates class 1 while a predicted probability < 0.5 indicates class 0
pred_classes = [1 if prob>0.5 else 0 for prob in np.ravel(predictions)]

"""We can look at a correlation matrix to show us what our model got right, and what it got wrong. """

from sklearn.metrics import confusion_matrix
import seaborn as sns

cf_matrix = confusion_matrix(test_labels, pred_classes)
cf_matrix = sns.heatmap(cf_matrix, annot=True, cmap="autumn", fmt='g')
cf_matrix.set_xlabel("Predicted Label", fontsize = 14)
cf_matrix.set_ylabel("True Label", fontsize = 14)

"""The model preformed fairly well on the test set. It was able to correctly classify 17543 samples. The main issue with the model is that we have 1350 false positives which are likely due to the class weights and bias that we initialized. Overall the accuracy on the test set was ~93%.

*Note: although seeds have been set, results may vary slightly.*

##**Scoring**

Finally, we will score the population dataset using our model.
"""

score_df = pd.read_csv("/content/drive/Shareddrives/The Mail Order Miracle/Data/9.8 Population v1.91.csv")

"""We will need to do a little bit of data cleaning before we can generate predictions.

Because we dropped the column WebUse when creating the model due to a significant proportion of null values, we will drop this column again.
"""

score_df = score_df.drop("WebUseBin", axis=1)

"""Any values that are null are filled using the same imputation process as above."""

from sklearn.impute import KNNImputer

cols_to_imputate = ['FinalAge', 'homeownr', 'ProdCatA_Bin']
impute = score_df[cols_to_imputate]

for col in impute:

  imputer = KNNImputer(n_neighbors=2)
  arr = np.array(impute[col])
  arr = arr.reshape(-1,1)
  arr = imputer.fit_transform(arr)
  score_df[col] = arr

"""Any non-numeric columns are made numeric using the same process as above."""

from sklearn import preprocessing

non_numeric = ["AgeCode", "gender", "homeval", "incgrp", "maritalgrp", "NoLonger", "region", "state", "UseNet"]

label_encoder = preprocessing.LabelEncoder()

for col in non_numeric:
  score_df[col] = label_encoder.fit_transform(score_df[col])

"""Once again we will take the log of ZLabel."""

scoring = score_df.copy()
#scoring['ZLab'] = np.log(scoring.pop('ZLabel'))
scoring.drop("ZLabel", axis=1, inplace=True)

"""Next we normalize the data."""

scoring = scaler.fit_transform(scoring)
scoring = scaler.transform(scoring)

scoring = np.clip(scoring, -5, 5)

"""Finally, we convert our data into an array."""

get_predictions = np.array(scoring)
get_predictions.shape

"""Now we can make our predictions on the population data."""

pop_predictions = model.predict([get_predictions])

"""We will assign our predictions to a column named "Predictions" in the population dataframe."""

score_df["Predictions"] = pop_predictions
score_df["Predictions"] = score_df.Predictions.round(decimals=5)

"""Our results are 30000 samples with the largest probabilities."""

results = score_df.nlargest(30000, "Predictions")
results

"""Lastly, we will save our results in an excel file."""

import os

results = results[["ZLabel", "Predictions"]]

os.chdir("/content/drive/Shareddrives/The Mail Order Miracle/Data")
file_name = "case9_scoring_results.xlsx"
results.to_excel(file_name, index=False)

import pandas as pd

df = pd.read_excel("/content/drive/Shareddrives/The Mail Order Miracle/Data/case9_scoring_results.xlsx", engine = "openpyxl")
df.hist(column="Predictions")