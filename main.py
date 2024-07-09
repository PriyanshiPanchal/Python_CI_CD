import os
import pandas as pd
import boto3
import io
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

# Global variables for model, scaler, and label encoder dictionary
model = None
scaler = None
le_dict = None

# Function to preprocess data
def preprocess_data(df, le_dict=None):
    df = df.copy()
    le_dict = le_dict or {}
    for col in df.columns:
        if df[col].dtype == 'object':
            if col not in le_dict:
                le_dict[col] = LabelEncoder()
                df[col] = le_dict[col].fit_transform(df[col])
            else:
                df[col] = le_dict[col].transform(df[col])
    return df, le_dict

# Function to train the model
def train_model(df):
    df, le_dict = preprocess_data(df)
    x = df.drop(['price'], axis=1)
    y = df['price']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    
    mmscaler = MinMaxScaler(feature_range=(0, 1))
    x_train = mmscaler.fit_transform(x_train)
    x_test = mmscaler.transform(x_test)
    
    modelETR = ExtraTreesRegressor()
    modelETR.fit(x_train, y_train)
    
    return modelETR, mmscaler, le_dict

# Function to predict prices
def predict_prices(model, scaler, le_dict, new_data):
    new_data_preprocessed, _ = preprocess_data(new_data, le_dict)
    new_data_scaled = scaler.transform(new_data_preprocessed)
    predictions = model.predict(new_data_scaled)
    
    return predictions

# Function to read data from S3
def read_s3_file(bucket_name, file_key, aws_access_key_id, aws_secret_access_key, region_name):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read()
        df = pd.read_csv(io.BytesIO(file_content))
        return df
    except s3_client.exceptions.NoSuchKey:
        print(f"Error: The file {file_key} does not exist in bucket {bucket_name}.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Initialization function to load and train model
def initialize_model():
    global model, scaler, le_dict
    
    bucket_name = 'smart-city-project'
    file_key = 'flight-prediction/Actual_DataSource.csv'
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region_name = 'ap-south-1'

    df = read_s3_file(bucket_name, file_key, aws_access_key_id, aws_secret_access_key, region_name)
    df.drop(columns=['flight', 'duration'], inplace=True)
    if df is not None:
        model, scaler, le_dict = train_model(df)
    else:
        model, scaler, le_dict = None, None, None

# Initialize the model when the script is first loaded
initialize_model()

# Main function for Google Cloud Function
def main(request):
    global model, scaler, le_dict
    
    # Ensure model is loaded properly
    if model is None or scaler is None or le_dict is None:
        return json.dumps({'error': 'Model is not trained properly.'}), 500, {'Content-Type': 'application/json'}
    
    # Handle POST request
    if request.method == 'POST':
        try:
            data = request.get_json()
            new_data = pd.DataFrame([data])
            
            # Predict prices
            predictions = predict_prices(model, scaler, le_dict, new_data)
            
            # Return predictions as JSON response
            return json.dumps({'predicted_prices': predictions.tolist()}), 200, {'Content-Type': 'application/json'}
        
        except Exception as e:
            return json.dumps({'error': str(e)}), 500, {'Content-Type': 'application/json'}
    
    # Handle other HTTP methods
    else:
        return json.dumps({'error': 'Unsupported HTTP method. Use POST Method.'}), 405, {'Content-Type': 'application/json'}

# def hello_world(request):
#     return "Hello, World!"