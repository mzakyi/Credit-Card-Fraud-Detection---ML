# -*- coding: utf-8 -*-
**"Credit Card Fraud Detection - ML.ipynb**

# Credit Card Fraud Detection - ML

###The goal of this project is to develop a machine learning model that can accurately detect fraudulent credit card transactions using historical data. By analyzing transaction patterns, the model should be able to distinguish between normal and fraudulent activity, helping financial institutions flag suspicious behavior early and reduce potential risks.

**Challenges include:**

Handling imbalanced datasets where fraud cases are a small fraction of total transactions.
Ensuring high precision to minimize false positives (flagging a valid transaction as fraud).
Ensuring high recall to detect as many fraud cases as possible.

## Step 1: Importing necessary Libraries

# Importing the necessary Python libraries needed for this project.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

**"""# Step 2: Loading the Data
Loading the dataset into a pandas DataFrame, examining its structure, and exploring the data we are dealing with.
**

"""

data = pd.read_csv("/content/creditcard.csv")

data

data.describe()

data.shape
# Data has (77338 rows & 31 columns

data.info()

"""### Understanding what the columns represent and their object type
1. Time: This shows how many seconds have passed since the first transaction in the dataset. (Integer)
2. V1-V28: These are special features created to hide sensitive information about the original data.(Float)
3. Amount: Transaction amount. (Float)
4. Class: Target variable (0 for normal transactions, 1 for fraudulent transactions). (Float)

# Cleaning our DataFrame
"""

# Let's see if we have any duplicates.
data[data.duplicated()]

# Lets drop our duplicates()
data = data.drop_duplicates()

# Let's check to make sure duplicates were dropped
data[data.duplicated()]

# We can see there are no null values in our dataframe
print(data.isnull().sum())

** #Step 3: Analyzing Class Distribution**

### The next step is to check the distribution of fraudulent vs. normal transactions.

Let's separate the dataset into two groups: fraudulent transactions (Class == 1) and valid transactions (Class == 0).
But first, let's convert the class column into an integer
"""

# But first lets convert the values in class to 1s(Fraudulent) and 0s(Valid Transactions)
data["Class"] = [1 if value==1.0 else 0 for value in data["Class"]]

#Checking to make sure our Class column is converted properly.
data

# Let's see how many fraudulent and valid cases are present in our data
fraud = data['Class'] == 1
valid = data['Class'] == 0
print('Fraud Cases:',len(fraud))
print('Valid Transactions:',len(valid))

"# Step 5: Plotting Correlation Matrix
Let's visualize the correlation between features using a heatmap and a correlation matrix. This will give us an understanding of how the different features are correlated and which ones may be more relevant for prediction.



"""

corr_matrix = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corr_matrix, vmax = .8, cmap="coolwarm", square = True)
plt.show()

#Most features do not correlate strongly with others, but some features, like V2 and V5, have a negative correlation with the Amount feature.
#This provides valuable insights into how the features are related to the transaction amounts.

# Step 6: Preparing Data
## Separating the features into  input features (X) and target variable (Y) before splitting the data into training and testing sets

X = data.drop(['Class'], axis = 1) removes the target column (Class) from the dataset to keep only the input features.
Y = data["Class"] selects the Class column as the target variable (fraud or not).

train_test_split(...) splits the data into training and testing sets, with 80% for training and 20% for testing.
random_state=42 ensures reproducibility (same split every time you run it).


X = data.drop(['Class'], axis = 1)
y = data["Class"]

# We import train_test_split from scikit learn to train, test, and split our data before we build our model
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 42)

**"#Step 7: Building and Training the Model
###First, we train a Random Forest Classifier to predict fraudulent transactions.

###from sklearn.ensemble import RandomForestClassifier This imports the RandomForestClassifier from sklearn.ensemble, which is used to create a random forest model for classification tasks.
**
"""

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier() #rfc = RandomForestClassifier(): Initializes RandomForestClassifier instance.
rfc.fit(XTrain, yTrain) #Trains the RandomForestClassifier model on(XTrain and yTrain)

yPred = rfc.predict(XTest) #Predicting the target labels for the test data (XTest)

# Step 8: Evaluating the Model
**After training the model, we need to evaluate its performance using various metrics such as accuracy, precision, recall, F1-score, and the Matthews correlation coefficient.
"""**

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
accuracy = accuracy_score(yTest, yPred)
precision = precision_score(yTest, yPred)
recall = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
mcc = matthews_corrcoef(yTest, yPred)

**#Printing out the results of our metrics**
print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

**"# In Conclusion:**
I observed that for:
Accuracy: Out of all predictions, 99.96% were correct. However, in imbalanced datasets (like fraud detection), accuracy can be misleading, i.e., a model that predicts everything as "not fraud" will still have high accuracy.

Precision: When the model predicted "fraud", it was correct 98.73% of the time. High precision means very few false alarms (false positives).

Recall: Out of all actual fraud cases, the model detected 79.59%. This shows how well it catches real fraud. A lower recall means some frauds were missed (false negatives).

F1-Score: A balance between precision and recall. 88.14% is strong and shows the model handles both catching fraud and avoiding false alarms well.

Matthews Correlation Coefficient (MCC): 0.8863: A more balanced score (from -1 to +1) even when classes are imbalanced. A value of 0.8863 is very good; it means the model is making strong, balanced predictions overall.

We can balance the dataset by oversampling the minority class or by undersampling the majority class, and we can increase the accuracy of our model.
"""
