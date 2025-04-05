import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import RFE



df = pd.read_csv("heart/heart.csv")

#Check for missing data
missing_values = df.isnull().sum()
#print(df.head())
#print(missing_values) # no missing data

#count # of duplicate rows and remove duplicates
df.describe()
duplicate_rows = df.duplicated()
print(f"Number of duplicate rows: {duplicate_rows.sum()}")

remove_duplicates = df.drop_duplicates()
print("Shape of DataFrame Before Removing Duplicates: ", df.shape)
print("Shape of DataFrame After Removing Duplicates: ", remove_duplicates.shape)
print(remove_duplicates.columns)

#separating categorical and numerical columns
num_columns = ['age', 'trestbps','chol','thalach', 'oldpeak','ca']
cat_columns = ['sex', 'cp','fbs','restecg','exang','slope','thal']

column_name = {
    'age': 'Age',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Serum Cholesterol',
    'thalach': 'Maximum Heart Rate Achieved',
    'oldpeak': 'Oldpeak (ST Depression)',
    'ca': 'Number of Major Vessels',
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting Electrocardiographic',
    'exang':'Exercise-Induced Angina',
    'slope':'Slope of Peak Exercise ST Segment',
    'thal':'Thalassemia',

}

#Z-score for numerical values, one-hot encoding for categorical
preprocess = ColumnTransformer(transformers=[
    ('number', StandardScaler(), num_columns),
    ('categories', OneHotEncoder(drop='first', sparse_output=False), cat_columns)
])

#Applying preprocessing
process_data_no_duplicate = preprocess.fit_transform(remove_duplicates)
process_data = preprocess.fit_transform(df)

cat_names = preprocess.transformers_[1][1].get_feature_names_out(cat_columns)
all_column = num_columns + list(cat_names)

#dataframe with processed data
processed_data_df = pd.DataFrame(process_data, columns=all_column)
no_dup_processed_data_df = pd.DataFrame(process_data_no_duplicate, columns=all_column)

x = df.drop(columns=['target'])
y = df['target']

x_process = preprocess.fit_transform(x)

model = RandomForestRegressor(random_state=12)
rfe = RFE(model, n_features_to_select=10)
x_selected = rfe.fit_transform(x_process, y)

selected_features = np.array(all_column)[rfe.support_]
print("Selected Features:", selected_features)

feature_importance = rfe.estimator_.feature_importances_
selected_features = np.array(all_column)[rfe.support_]

sorted = np.argsort(feature_importance)[::-1]
sorted_feature = selected_features[sorted]
sorted_importance = feature_importance[sorted]

#Bar plot to display the important features
plt.figure(figsize=(10,6))
sns.barplot(x=sorted_importance, y=sorted_feature, palette="coolwarm")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Feature")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(process_data, df['target'], test_size=0.2, random_state=42)
print(f"Training Data Shape: {x_train.shape}, Test Data Shape: {x_test.shape}")

#visualization for numerical data
#Histogram (to see distribution)
plt.figure(figsize=(12,8))
for i, col in enumerate(num_columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {column_name[col]}')
    plt.xlabel(column_name[col])
plt.tight_layout()
plt.show()

#visualization for categorical data
#Bar chart (compare the amount of different types)
plt.figure(figsize=(12,8))
for i, col in enumerate(cat_columns, 1):
    plt.subplot(2, 4, i)
    sns.countplot(x=df[col])
    plt.title(f'{column_name[col]}')
    plt.xlabel(column_name[col])
plt.tight_layout()
plt.show()


#visualization for coorelation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[num_columns].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap")
plt.show()
