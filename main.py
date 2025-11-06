import os
import sys
import numpy as np
import torch
from tensorflow.keras.models import load_model
from tabnet import TabNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import cv2

#load models 
tabnet_model = "models/tabnet_heartattack_best.pt"
cnn_model = "models/cnn_braintumor_best.h5"

num_columns = ['age', 'trestbps','chol','thalach', 'oldpeak','ca']
cat_columns = ['sex', 'cp','fbs','restecg','exang','slope','thal']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_heart_attack(csv_path):
    print(f" Processing health data from: {csv_path}")
    df = pd.read_csv(csv_path)

    #check which data are contained
    available_columns = df.columns.tolist()
    contain_num = [a for a in num_columns if a in available_columns]
    contain_cat = [a for a in cat_columns if a in available_columns]

    if not contain_num and not contain_cat:
        print("data in this csv file is not valid")
        return

    #Z-score for numerical values, one-hot encoding for categorical
    preprocess = ColumnTransformer(transformers=[
        ('number', StandardScaler(), contain_num),
        ('categories', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), contain_cat)
    ], remainder='drop')

    x = df[contain_num + contain_cat]
    x_process = preprocess.fit_transform(x)
    x_tensor = torch.tensor(x_process, dtype=torch.float32).to(device)

    #run tabnet
    model = TabNet(inp_dim=x_process.shape[1], final_out_dim=1).to(device)
    model.load_state_dict(torch.load(tabnet_model, map_location=device))
    model.eval()

    with torch.no_grad():
        predictions, _ = model(x_tensor)
    predictions = torch.sigmoid(predictions).cpu().numpy().flatten()

    print("\n Heart Attack Risk Predictions:")
    for i, p in enumerate(predictions):
        print(f" Sample {i+1}: {p:.3f}")

#loading mri image
def load_image(path, target_size=(256, 256)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image from: {path}")
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=(0, -1))

#running brain tumor model
def diagnose_tumor(image_path):
    print(f"\n The input MRI image: {os.path.basename(image_path)}")
    model = load_model(cnn_model)
    img = load_image(image_path)
    predictions = model.predict(img)
    label = np.argmax(predictions, axis=1)[0]
    print(f" Predicted tumor type: {label}")
    print(f" Probabilities: {np.round(predictions[0], 3)}")


#main method which checks what type of input data it is
def main(input_path):
    if os.path.isdir(input_path):

        #check file type
        files = os.listdir(input_path)
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if csv_files and image_files:
            print(f"The input files includes both csv file and MRI images")
            for csv_file in csv_files:
                predict_heart_attack(os.path.join(input_path, csv_file))
            for image_file in image_files:
                diagnose_tumor(os.path.join(input_path, img_file))

        elif csv_files:
            print(f"Input data only contains csv data. \nNow running heart attack prediction")
            for csv_file in csv_files:
                predict_heart_attack(os.path.join(input_path, csv_file))

        elif image_files:
            print(f"Input only contains MRI images. \nNow running brain tumor detection")
            for img_file in image_files:
                diagnose_tumor(os.path.join(input_path, img_file))

        else:
            print("The inputted data does not contain any csv files or mri images")

    #only contain a single csv file
    elif input_path.lower().endswith('.csv'):
        print("There is only a single csv file. \n Now running heart attack prediction")
        predict_heart_attack(input_path)
    elif input_path.lower().endswith(('.png', '.jpg','.jpeg')):
        print("There are only images in this folder. \n Now running brain tumor detection")
        diagnose_tumor(input_path)
    else:
        print("Does not support the file inputted. Please input csv files, or images")

if __name__== "__main__":
    if len(sys.argv) < 2:
        print("python main.py <input_path>")
    else:
        main(sys.argv[1])