#!/usr/bin/env python
# coding: utf-8

# ## Business Objective

# Prediction of overall disease according to behavioral and medicine

# ## Diabetes prediction

# In[171]:


# library for data manipulation and analysis
import pandas as pd  
import numpy as np  
import sklearn   

# data visualization
import seaborn as sns  
import matplotlib.pyplot as plt  
import matplotlib as matplot  

# import re  # module for regular expression operations

# do not display warnings
import warnings  
warnings.filterwarnings("ignore")  


# ### Dataset Merge & select attribute

# In[172]:


df1 = pd.read_csv('./input/labs.csv')
df2 = pd.read_csv('./input/examination.csv')
df3 = pd.read_csv('./input/demographic.csv')
df4 = pd.read_csv('./input/diet.csv')
df5 = pd.read_csv('./input/questionnaire.csv')

df2.drop(['SEQN'], axis = 1, inplace=True)
df3.drop(['SEQN'], axis = 1, inplace=True)
df4.drop(['SEQN'], axis = 1, inplace=True)
df5.drop(['SEQN'], axis = 1, inplace=True)

df = pd.concat([df1, df2], axis=1, join='inner')
df = pd.concat([df, df3], axis=1, join='inner')
df = pd.concat([df, df4], axis=1, join='inner')
df = pd.concat([df, df5], axis=1, join='inner')

df.describe()


# In questionnaire.csv, DIQ010 and DID040 are both diabetes-related data. 
# We need to decide which one to use.
# 
# Decide to use DIQ010 (it asks if the person has ever been diagnosed with diabetes by a doctor).

# In[173]:


count_A = df['DIQ010'].value_counts() # Has a doctor ever diagnosed you with diabetes 2: No, 1: Yes (presumably)
                                      # The meaning of the rest of the data is unknown(3, 7, 9).
print(count_A)


# In[174]:


count_A = df['DID040'].value_counts() # Age at diabetes diagnosis
print(count_A)


# ### Select the data (columns) to use

# In[175]:


from sklearn.feature_selection import VarianceThreshold

df.dropna(axis=1, how='all')
df.dropna(axis=0, how='all')

df = df.rename(columns = {'SEQN' : 'ID', # ID
                          'RIAGENDR' : 'Gender', # Gender
                          'INDFMPIR' : 'Family_income', # Income
                          'LBXGH' : 'GlycoHemoglobin', # Glycated hemoglobin
                          'BMXARMC' : 'ArmCircum', # Arm circumference
                          'BMDAVSAD' : 'SaggitalAbdominal', # Abdominal circumference
                          'MGDCGSZ': 'GripStrength', # grip strength
                          'ALQ130' : 'Alcohol', # number of alcohol consumption in a week
                          'SLD010H' : 'Sleep_time', # sleep time
                          'SMQ040' : 'Smoke', # Smoking status
                          'PAQ715' : 'Video_game', # Time spent playing video games
                          'WHD010':'Height', # Height
                          'WHD020':'Weight', # Weight 
                          'MCQ300B' : 'Family_history_of_asthma', # Family history of asthma
                          'MCQ300C' : 'Family_history_of_diabetes', # Family history of diabetes
                          'DIQ010' : 'Diabetes'}) # Diabetes

df = df.loc[:, ['ID', 'Gender', 'Family_income', 'ArmCircum', 
                'SaggitalAbdominal', 'GripStrength', 'Alcohol', 'Sleep_time','Smoke', 'Video_game', 'Height', 'Weight', 'Family_history_of_asthma', 'Family_history_of_diabetes', 'GlycoHemoglobin', 'Diabetes']]

df.describe()


# ### Data Exploration

# In[176]:


# check how many vaule are missing
for column in df.columns:
    null_count = df[column].isnull().sum()
    print("{}: {}".format(column, null_count))


# In[177]:


import seaborn as sns
import matplotlib.pyplot as plt

# Get the number of columns in the DataFrame
num_columns = len(df.columns)

# Calculate the number of rows and columns for subplots
num_rows = (num_columns + 2) // 3
num_cols = min(num_columns, 3)

# Create subplots with the specified layout
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))

# Iterate over the columns and create box plots
for idx, column in enumerate(df.columns):
    ax = axs[idx // num_cols, idx % num_cols]
    sns.boxplot(x=df[column], ax=ax)
    ax.set_xlabel(column)
    # ax.set_title('Box Plot of {}'.format(column))

# Remove empty subplots if necessary
if num_columns % 3 != 0:
    for idx in range(num_columns, num_rows * num_cols):
        fig.delaxes(axs.flatten()[idx])

# Adjust the spacing between subplots
fig.tight_layout()

fig.suptitle("Data Exploration")

# Show the plots
plt.show()


# In[178]:


import seaborn as sns
import matplotlib.pyplot as plt

# Get the number of columns in the DataFrame
num_columns = len(df.columns)

# Calculate the number of rows and columns for subplots
num_rows = (num_columns + 2) // 3
num_cols = min(num_columns, 3)

# Create subplots with the specified layout
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))

# Iterate over the columns and create distribution plots
for idx, column in enumerate(df.columns):
    ax = axs[idx // num_cols, idx % num_cols]
    sns.distplot(df[column], kde=False, ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    # ax.set_title('Distribution of {}'.format(column))

# Remove empty subplots if necessary
if num_columns % 3 != 0:
    for idx in range(num_columns, num_rows * num_cols):
        fig.delaxes(axs.flatten()[idx])

# Adjust the spacing between subplots
fig.tight_layout()

fig.suptitle("Data Exploration")

# Show the plots
plt.show()


# See that Alcohol, Sleep_time, Height, and Weight have outliers.
# 
# smoke, Family_history_of_asthma, Family_history_of_diabetes, and Diabetes should only have 2 values because they are Yes/No responses.

# 

# ## Preprocessing

# In[179]:


df.describe()


# In[180]:


# we are creating a model from survey data -> the sincerity of the respondents affects the reliability of the model
# If a response has more than 4 blanks, drop that row

df.dropna(thresh=df.shape[1]-4, inplace=True)


# In[181]:


df.describe()


# In[182]:


# handle outliers

df = df[np.logical_or(df['Alcohol'] < 100, df['Alcohol'].isna())] # drop if value > 100
df = df[np.logical_or(df['Height'] < 400, df['Height'].isna())] # drop if value > 400
df = df[np.logical_or(df['Weight'] < 400, df['Weight'].isna())] # drop if value > 400
df = df[np.logical_or(df['Sleep_time'] < 20, df['Sleep_time'].isna())] # drop if value > 20


df = df[df['Family_history_of_asthma'] < 3] # drop if value > 3 (1, leaving only 2)
df = df[df['Family_history_of_diabetes'] < 3] # drop if value > 3 drop (1, keep 20)
df = df[df['Diabetes'] < 3] # drop if value > 3 drop (1, 20k left)


# 0: not, 1: yes
# mapping as 1->0, 2->1
mapping_dict = {1: 0, 2: 1}
df['Gender'] = df['Gender'].map(mapping_dict)
df['Family_history_of_asthma'] = df['Family_history_of_asthma'].map(mapping_dict)
df['Family_history_of_diabetes'] = df['Family_history_of_diabetes'].map(mapping_dict)
df['Diabetes'] = df['Diabetes'].map(mapping_dict)

df['Smoke'] = df['Smoke'].map({1: 0, 2: 1, 3: 1})


df.describe()


# ### Handle missing vaule

# In[183]:


# Check how many vaule are missing
for column in df.columns:
    null_count = df[column].isnull().sum()
    print("{}: {}".format(column, null_count))


# In[184]:


# fill in missing vaule
# Since categorical data has been processed above
# the remaining data is numerical data
# so fill the missing value with median
  
df['Family_income'] = df['Family_income'].fillna(df['Family_income'].median())
df['SaggitalAbdominal'] = df['SaggitalAbdominal'].fillna(df['SaggitalAbdominal'].median())
df['Alcohol'] = df['Alcohol'].fillna(df['Alcohol'].median())
df['Sleep_time'] = df['Sleep_time'].fillna(df['SaggitalAbdominal'].median())
df['ArmCircum'] = df['ArmCircum'].fillna(df['ArmCircum'].median())
df['GripStrength'] = df['GripStrength'].fillna(df['GripStrength'].median())
df['Height'] = df['Height'].fillna(df['Height'].median())
df['Weight'] = df['Weight'].fillna(df['Weight'].median())


# Smoke is filled with ffill
df['Smoke'] = df['Smoke'].fillna(method='ffill')


# ### Validate which model to use to populate GlycoHemoglobin

# In[185]:


import matplotlib.pyplot as plt

# Specify the column to visualize
column_name = 'GlycoHemoglobin'

# Plot a histogram
plt.hist(df[column_name], bins=20)
plt.xlabel(column_name)
plt.ylabel('Frequency')
plt.title('Distribution of ' + column_name)
plt.show()


# In[186]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Define the target and feature columns
target_column = 'GlycoHemoglobin'
feature_columns = [column for column in df.columns if column != target_column and column != 'Diabetes']

# Split the data into missing and non-missing target values
df_missing = df[df[target_column].isnull()]
df_not_missing = df[df[target_column].notnull()]

# Prepare the training data
X_train = df_not_missing[feature_columns]
y_train = df_not_missing[target_column]

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=423)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the regressor models
regressors = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor()
}

# Perform cross-validation and evaluate performance metrics for each regressor
results = {}
for name, regressor in regressors.items():
    regressor.fit(X_train, y_train)  # Train the regressor on the training data
    y_pred = regressor.predict(X_test)  # Make predictions on the testing data
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = regressor.score(X_test, y_test)  # Compute the accuracy score
    results[name] = {'MSE': mse, 'MAE': mae, 'R-squared': r2, 'Accuracy': accuracy}

# Display the performance results
performance_df = pd.DataFrame(results)
print(performance_df)


# In[187]:


x = np.linspace(0, len(y_test), len(y_test))

# Plot the actual values
plt.figure(figsize=(12, 8))
plt.scatter(x, y_test, color='blue', label='Actual Values')

# Plot the model predictions
for name, regressor in regressors.items():
    regressor.fit(X_train, y_train)  # Train the regressor on the training data
    y_pred = regressor.predict(X_test)  # Make predictions on the testing data
    plt.plot(x, y_pred, label=name)

plt.xlabel('Data Points')
plt.ylabel('GlycoHemoglobin')
plt.title('Actual Values vs Model Predictions')
plt.legend()
plt.show()


# ## Populate glycohemoglobin with a linear regression analyzer

# In[188]:


from sklearn.linear_model import LinearRegression

# Function to fill missing values ​​using linear regression
def fill_missing_with_linear_regression(df, target_column):
    # Separate the data into rows with and without missing values ​​in target_column
    df_missing = df[df[target_column].isnull()]
    df_not_missing = df[df[target_column].notnull()]

    # Prepare feature and target data to train linear regression
    feature_columns = []
    for column in df.columns:
        if column != target_column and column != 'Diabetes': # Exclude target column and add to feature_columns
            feature_columns.append(column)
    
    X_train = df_not_missing[feature_columns]
    y_train = df_not_missing[target_column]

    # Linear regression training
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predict using linear regression to fill in missing values
    X_missing = df_missing[feature_columns]
    y_missing = regressor.predict(X_missing)

    # Fill missing values ​​with predicted values
    df.loc[df[target_column].isnull(), target_column] = y_missing

    return df

# Specify column to fill in missing value
target_column = 'GlycoHemoglobin'

# Fill in missing values ​​using linear regression
df = fill_missing_with_linear_regression(df, target_column)


# In[189]:


# Check for missing vaule
for column in df.columns:
    null_count = df[column].isnull().sum()
    print("{}: {}".format(column, null_count))


# In[190]:


import seaborn as sns
import matplotlib.pyplot as plt

# Get the number of columns in the DataFrame
num_columns = len(df.columns)

# Calculate the number of rows and columns for subplots
num_rows = (num_columns + 2) // 3
num_cols = min(num_columns, 3)

# Create subplots with the specified layout
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))

# Iterate over the columns and create box plots
for idx, column in enumerate(df.columns):
    ax = axs[idx // num_cols, idx % num_cols]
    sns.boxplot(x=df[column], ax=ax)
    ax.set_xlabel(column)
    # ax.set_title('Box Plot of {}'.format(column))

# Remove empty subplots if necessary
if num_columns % 3 != 0:
    for idx in range(num_columns, num_rows * num_cols):
        fig.delaxes(axs.flatten()[idx])

# Adjust the spacing between subplots
fig.tight_layout()

fig.suptitle("Data Exploration")

# Show the plots
plt.show()


# In[191]:


import seaborn as sns
import matplotlib.pyplot as plt

# Get the number of columns in the DataFrame
num_columns = len(df.columns)

# Calculate the number of rows and columns for subplots
num_rows = (num_columns + 2) // 3
num_cols = min(num_columns, 3)

# Create subplots with the specified layout
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))

# Iterate over the columns and create distribution plots
for idx, column in enumerate(df.columns):
    ax = axs[idx // num_cols, idx % num_cols]
    sns.distplot(df[column], kde=False, ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    # ax.set_title('Distribution of {}'.format(column))

# Remove empty subplots if necessary
if num_columns % 3 != 0:
    for idx in range(num_columns, num_rows * num_cols):
        fig.delaxes(axs.flatten()[idx])

# Adjust the spacing between subplots
fig.tight_layout()

fig.suptitle("Data Exploration")

# Show the plots
plt.show()


# In[192]:


colormap = plt.cm.viridis  # Set the colormap to viridis

plt.figure(figsize=(15,15)) 

# Create a heatmap to visualize the correlation matrix of the DataFrame.
sns.heatmap(df.astype(float).drop(axis=1, labels='ID').corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, annot=True) 


# In[193]:


#data -> attributes, target -> diabetes
X = df.iloc[:,:-1]

y = df.iloc[:,-1]
y.describe()


# ## Over sampling diabetes data using SMOTE

# In[194]:


from imblearn.over_sampling import SMOTE
X, y = SMOTE().fit_resample(X, y)


# In[195]:


pd.Series(y).value_counts().plot(kind='bar', title='Class distribution after appying SMOTE')


# # Create a full-fledged diabetes prediction model
# 

# In[196]:


from sklearn import linear_model # Importing the linear_model module from the sklearn library
from sklearn.preprocessing import StandardScaler  # Importing StandardScaler from sklearn.preprocessing to standardize features

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=423)

# Create an instance of the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform the training data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create an instance of the Linear Regression model
lr_regr = linear_model.LinearRegression()

# Fit the linear regression model to the training data
lr_regr.fit(X_train, y_train)

# Make predictions using the test set
lr_pred_diabetes = lr_regr.predict(X_test)

# Check the performance of the model using the score method which returns the coefficient of determination R^2 of the prediction
# The best possible score is 1.0 for R^2
lr_score = lr_regr.score(X_test, y=y_test)

print('LR_Coefficients: ', lr_regr.coef_)  # Print the coefficients learnt by the regression model
print('LR_Mean Square Error: %.2f' % mean_squared_error(y_test, lr_pred_diabetes))  # Print the mean squared error of the prediction
print('LR_Variance score: %.2f' % r2_score(y_test, lr_pred_diabetes))  # Print the R^2 score of the prediction, which tells us the percentage of the variance in the target variable that is predictable from the features
print('Score: %.2f' % lr_regr.score(X_test, y_test))  # Print the R^2 score again


# In[197]:


from sklearn.ensemble import AdaBoostClassifier  # Importing AdaBoostClassifier from sklearn.ensemble
from sklearn.tree import DecisionTreeClassifier  # Importing DecisionTreeClassifier from sklearn.tree

# Create an AdaBoost classifier object. 

ab_clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(),  # Set the base estimator to a decision tree.
    n_estimators=100,  # The number of models to iteratively train.
    learning_rate=0.5,  # How much to shrink the contribution of each classifier after it's trained.
    random_state=100  # The seed used by the random number generator for reproducibility.
)

# Training the AdaBoost classifier with the training data
ab_clf.fit(X_train, y_train)
print("training....\n")

# Making predictions using the trained AdaBoost classifier on the test data
ab_pred_diabetes = ab_clf.predict(X_test)
print('prediction: \n', ab_pred_diabetes)

# Print the parameters of the AdaBoost classifier
print('\nparms: \n', ab_clf.get_params)

# Calculate and print the staged scores for the AdaBoost classifier. 
# This will output the mean accuracy at each stage of the boosting process.
staged_scores = list(ab_clf.staged_score(X_test, y_test))
print('Staged scores: ', staged_scores)

# Calculate and print the mean accuracy of the AdaBoost classifier.
# This will output the mean accuracy of the classifier on the test data.
ab_clf_score = ab_clf.score(X_test, y_test)
print("\nmean accuracy: %.2f" % ab_clf.score(X_test, y_test))


# In[198]:


from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[199]:


# Instantiate a BaggingClassifier
bagging = BaggingClassifier(
    base_estimator=KNeighborsClassifier(),  # Set the base estimator to a K-nearest neighbors classifier.
    max_samples=0.5,  # The number or fraction of samples to draw from X to train each base estimator.
    max_features=0.5,  # The number or fraction of features to draw from X to train each base estimator.
    bootstrap=False,  # Whether samples are drawn with replacement.
    bootstrap_features=False  # Whether features are drawn with replacement.
)

# Train the BaggingClassifier on the training data
bagging.fit(X_train, y_train)

# Use the trained BaggingClassifier to make predictions on the test data
bg_pred_diabetes = bagging.predict(X_test)

# Calculate the mean accuracy of the BaggingClassifier on the test data
bg_dt_score = bagging.score(X_test, y_test)
print('Mean accuracy of Bagging Classifier: ', bagging.score(X_test, y_test))


# In[200]:


# Instantiate a BaggingClassifier
bagging = BaggingClassifier(
    base_estimator=KNeighborsClassifier(),  # Set the base estimator to a K-nearest neighbors classifier.
    max_samples=0.5,  # The number or fraction of samples to draw from X to train each base estimator.
    max_features=0.5,  # The number or fraction of features to draw from X to train each base estimator.
    bootstrap=False,  # Whether samples are drawn with replacement.
    bootstrap_features=False  # Whether features are drawn with replacement.
)

# Train the BaggingClassifier on the training data
bagging.fit(X_train, y_train)

# Use the trained BaggingClassifier to make predictions on the test data
bg_pred_diabetes = bagging.predict(X_test)

# Calculate the mean accuracy of the BaggingClassifier on the test data
bg_score = bagging.score(X_test, y_test)
print('Mean accuracy of Bagging Classifier: ', bagging.score(X_test, y_test))


# In[201]:


# Instantiate a GradientBoostingClassifier.
boosting = GradientBoostingClassifier(
    max_depth=3,  # The maximum depth of the individual regression estimators.
    n_estimators=100,  # The number of boosting stages to perform.
    learning_rate=0.1  # Determines the impact of each tree on the final outcome.
)

# Fit the GradientBoostingClassifier to the training data
boosting.fit(X_train, y_train)

# Use the trained GradientBoostingClassifier to make predictions on the test data
boost_pred_diabetes = boosting.predict(X_test)

# Calculate the mean accuracy of the GradientBoostingClassifier on the test data
boost_score = boosting.score(X_test, y_test)
print("Gradient Boosting Classifier 정확도:", boost_score)


# # Compare the accuracy of each model

# In[202]:


d = {'Model': ['Linear Regression', 'Adaboost', 'Bagging_decision tree based', 'Bagging_KNeighbors', 'Gradient Boosting'],
     'accuracy' : [lr_score, ab_clf_score, bg_dt_score, bg_score, boost_score]}

result_df = pd.DataFrame(data = d)
result_df


# 

# In[203]:


result_df.plot(x='Model', y='accuracy', kind='bar', figsize=(8, 8), title='Diabetes Prediction Accuracy', 
               sort_columns=True)


# ## Validate GradientBoostingClassifier using KFold.

# In[206]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Instantiate a GradientBoostingClassifier. This is a machine learning model that uses the Gradient Boosting 
# ensemble method. It operates by adding new models to the ensemble sequentially. Each new model 
# gradually minimizes the loss function of the whole system using the Gradient Descent method.
boosting = GradientBoostingClassifier(max_depth=3, n_estimators=100, learning_rate=0.1)

# Create a KFold object to split the data into 10 folds for cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform K-fold cross validation
scores = []
for train_index, test_index in kf.split(X):
    # Split the data into training and testing sets for this fold
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model on the training data and calculate the accuracy on the testing data
    boosting.fit(X_train, y_train)
    y_pred = boosting.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)

# Print the accuracy for each fold
for fold, score in enumerate(scores, 1):
    print(f"Fold {fold} accuracy: {score:.4f}")

# Calculate and print the average accuracy across all folds
mean_score = sum(scores) / len(scores)
print(f"Mean Accuracy: {mean_score:.4f}")

