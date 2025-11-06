import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report, r2_score
from sklearn.feature_selection import RFE
from math import sqrt
from sklearn.model_selection import train_test_split
from tabnet import TabNet
import os
import datetime


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
    ('categories', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_columns)
], remainder='drop')

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

def to_label(feature):
    for col in column_name:
        if feature.startswith(col):
            return column_name[col]
    return feature

sorted_feature = [to_label(feature) for feature in selected_features[sorted]]
sorted_importance = feature_importance[sorted]

x_train, x_test, y_train, y_test = train_test_split(process_data_no_duplicate, remove_duplicates['target'], test_size=0.2, random_state=42)
print(f"Training Data Shape: {x_train.shape}, Test Data Shape: {x_test.shape}")

#calculate euclidean distance for KNN
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

#retrieve neighbors for KNN
def get_neighbors(train, test_row, num_neighbors):
    distance_list = list()
    for train_row in train:
        distance = euclidean_distance(test_row, train_row)
        distance_list.append((train_row, distance))
    distance_list.sort(key=lambda tup: tup[1])
    neighbors = [distance_list[i][0] for i in range(num_neighbors)]
    return neighbors

#KNN
def predict_regression(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = sum(output_values) / len(output_values)
    return prediction



#split data without removing duplicate
train_data = np.hstack((x_train, np.array(y_train).reshape(-1,1)))

predictions = []
for row in x_test:
    prediction = predict_regression(train_data, row, num_neighbors=5)
    predictions.append(prediction)

predictions = np.array(predictions)

knn_mse = mean_squared_error(y_test, predictions)
knn_mae = mean_absolute_error(y_test, predictions)
knn_rmse = np.sqrt(knn_mse)

print(f"\nEvaluation of KNN Regression:")
print(f"Mean Absolute Error (MAE): {knn_mae:.4f}")
print(f"Mean Squared Error (MSE): {knn_mse:.4f}")
#print(f"Root Mean Squared Error (RMSE): {knn_rmse:.4f}")

index = 0
test_sample = x_test[index]
prediction = predict_regression(train_data, test_sample, num_neighbors=5)

#print(f"Predict value (regression): {prediction:.4f}")
#print(f"Actual value: {y_test.iloc[index]}")

#add accuracy
knn_r2 = r2_score(y_test, predictions)
print(f"R-squared Score:{knn_r2:.4f}")


x_train, x_test, y_train, y_test = train_test_split(process_data_no_duplicate, remove_duplicates['target'], test_size=0.2, random_state=42)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1,1), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.reshape(-1,1), dtype=torch.float32)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TabNet(inp_dim=x_train.shape[1], final_out_dim=1).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 50
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        preds, sparse_loss = model(batch_x)
        loss = criterion(preds, batch_y) + 1e-3 * sparse_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")


model.eval()
with torch.no_grad():
    preds, _ = model(x_test_tensor.to(device))
    preds_cpu = preds.cpu()
    y_test_cpu = y_test_tensor.cpu()

    tab_mse = mean_squared_error(y_test_cpu,  preds_cpu)
    tab_mae = mean_absolute_error(y_test_cpu, preds_cpu)
    tab_r2 = r2_score(y_test_cpu, preds_cpu)

    print(f"\nEvaluation of TabNet:")
    print(f"MAE score: {tab_mae:.4f}")
    print(f"MSE score: {tab_mse: .4f}")
    print(f"R-squared Score:{tab_r2:.4f}")

    print("Sample Predictions vs Actual:")
    for i in range(5):
        pred_val = preds_cpu[i].item()
        actual_val = y_test_cpu[i].item()
        print(f"Pred: {pred_val: .4f}, Actual: {actual_val: .4f}")

#saving the best trained model
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

#find path to best results
best_scores = os.path.join(models_dir, "best_tabnet.txt")
best_r2 = -float("inf")

#load last best result
if os.path.exists(best_scores):
    with open(best_scores, "r") as f:
        try:
            best_r2 = float(f.read().strip())
        except:
            best_r2 = -float("inf")

# if new is better than old results
if tab_r2 > best_r2:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_model = os.path.join(models_dir, f"tabnet_heartattack_{timestamp}.pt")
    torch.save(model.state_dict(), current_model)

    with open(best_scores, "w") as f:
        f.write(str(tab_r2))
    torch.save(model.state_dict(), os.path.join(models_dir, "tabnet_heartattack_best.pt"))

    print(f"\nNew best model saved ({current_model}) with R-squared: {tab_r2:.4f}")
else:
    print(f"\nNo improvement (Current R-Squared = {tab_r2: .4f}, Best = {best_r2:.4f})")



#comparison visualization
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'MAE', 'R2 Score'],
    'KNN': [knn_mse, knn_mae, knn_r2],
    'TabNet': [tab_mse, tab_mae, tab_r2]
})

metrics_df = pd.melt(metrics_df, id_vars=['Metric'], var_name='Model', value_name='Score')

sns.set(style="whitegrid", palette="pastel")

plt.figure(figsize=(10,6))
sns.barplot(x="Metric", y='Score', hue="Model", data=metrics_df)
plt.title('KNN vs TabNet Performance', fontsize=16)
plt.ylabel('Score', fontsize= 14)
plt.xlabel('Metric', fontsize=14)
plt.legend(title='Model')
plt.tight_layout()
#plt.show()