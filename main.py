import pandas as pd
import json
import os
import warnings
import sklearn
import pickle
import firebase_admin
import requests
import numpy as np
import re
import random
from werkzeug.security import generate_password_hash, check_password_hash
from firebase_admin import credentials, auth, initialize_app, firestore, storage
from sklearn.svm import SVC
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, session, flash
from dotenv import load_dotenv
from datetime import datetime, timedelta
from data_processing import process_data
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from flask_mail import Mail, Message
from dateutil import parser
from twilio.rest import Client
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = 'secret'  

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'josephcollantes65@gmail.com'  # Replace with your Gmail address
app.config['MAIL_PASSWORD'] = 'oewu ulom kpax koqr'       # Replace with your Gmail App Password
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

account_sid = 'ACe3f45dff58eb11d6e018fc205bffc16b'
auth_token = '28ea2a902fb6d439254589319017f19a'

client = Client(account_sid, auth_token)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('C:/Users/Joseph Collantes/Desktop/THESIS/auth-key.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'dengue-prediction-d862e.appspot.com'  # Replace with your Firebase Storage bucket URL
})
db = firestore.client()
bucket = storage.bucket()

mail = Mail(app)

load_dotenv()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx'}

def read_file(file_path):
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path, engine='openpyxl')
        else:
            raise ValueError('Unsupported file format')
    except UnicodeDecodeError:
        if file_path.endswith('.csv'):
            try:
                return pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
            except Exception as e:
                raise ValueError(f"Error reading file with alternative encoding: {e}")
        else:
            raise

def convert_date_columns(df):
    date_columns = df.select_dtypes(include=['object']).columns[
        df.select_dtypes(include=['object']).apply(
            lambda x: x.str.match(r'\d{1,2}/\d{1,2}/\d{4}').any()
        )
    ]

    for col in date_columns:
        df[col] = pd.to_datetime(df[col], dayfirst=False, errors='coerce')

    for col in df.select_dtypes(include=['object']).columns:
        if col not in date_columns:
            df[col] = df[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.lower()

    return df
    
def upload_to_firebase(file, filename, folder_name):
    file.seek(0) 
    blob = bucket.blob(f'{folder_name}/{filename}')
    blob.upload_from_file(file, content_type=file.content_type)
    return blob.public_url
    
def set_no_cache(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

def doy_to_date(year, doy):
    return datetime(year, 1, 1) + timedelta(days=doy - 1)

def map_age_to_agegroup(age):
    if age >= 80:
        return '80+'
    elif 70 <= age <= 79:
        return '70-79'
    elif 60 <= age <= 69:
        return '60-69'
    elif 50 <= age <= 59:
        return '50-59'
    elif 40 <= age <= 49:
        return '40-49'
    elif 30 <= age <= 39:
        return '30-39'
    elif 20 <= age <= 29:
        return '20-29'
    elif 10 <= age <= 19:
        return '10-19'
    else:
        return '0-9'
    
# Define scoring function for each attribute
def get_risk_score(attribute, value):
    if attribute == 'Predicted_Cases':
        if value >= 7:
            return 4  # Severe/Critical Risk
        elif value >= 5:
            return 3  # High Risk
        elif value >= 3:
            return 2  # Moderate Risk
        elif value >= 1:
            return 1  # Low Risk
        else:
            return 0  # No Risk
    
    elif attribute == 't2m_max':  # Maximum Temperature
        # Based on research, temperatures between 25°C-35°C are ideal for Aedes mosquito survival and reproduction
        if value >= 35:
            return 4  # Severe/Critical Risk (High temperatures can drive high mosquito activity)
        elif value >= 30:
            return 3  # High Risk
        elif value >= 25:
            return 2  # Moderate Risk
        else:
            return 1  # Low Risk
            
    elif attribute == 'prectotcorr':  # Precipitation
        # High rainfall correlates with increased mosquito breeding sites due to stagnant water pools
        if value >= 15:
            return 4  # Severe/Critical Risk (Heavy rain increases mosquito breeding grounds)
        elif value >= 10:
            return 3  # High Risk
        elif value >= 5:
            return 2  # Moderate Risk
        else:
            return 1  # Low Risk
            
    elif attribute == 'ws2m_max':  # Maximum Wind Speed
        # Lower wind speeds allow mosquitoes to move more freely, facilitating disease spread
        if value <= 0.5:
            return 4  # Severe/Critical Risk (Very low wind; mosquitoes thrive)
        elif value <= 1:
            return 3  # High Risk
        elif value <= 2:
            return 2  # Moderate Risk
        else:
            return 1  # Low Risk
            
    elif attribute == 'rh2m':  # Relative Humidity
        # Humidity levels that support mosquito survival and breeding
        if value > 0.75:  # Above 75% (High Risk)
            return 3  # High Risk
        elif value >= 0.60:  # 60-75% (Moderate Risk)
            return 2  # Moderate Risk
        else:  # Below 60% (Low Risk)
            return 1  # Low Risk

            
    elif attribute == 'population_2020':  # Population
        if value > 3000:
            return 4  # Severe/Critical Risk
        elif value > 1000:
            return 3  # High Risk
        elif value > 500:
            return 2  # Moderate Risk
        else:
            return 1  # Low Risk
            
    elif attribute == 'average_household_size':  # Average Household Size
        if value >= 6:
            return 4  # Severe/Critical Risk
        elif value >= 5:
            return 3  # High Risk
        elif value >= 4:
            return 2  # Moderate Risk
        else:
            return 1  # Low Risk
            
    elif attribute == 'agegroup':  # Age Group
        if value in ['0-9', '70-79', '80+']:
            return 4  # Severe/Critical Risk
        elif value in ['10-19', '60-69']:
            return 3  # High Risk
        elif value in ['20-39']:
            return 2  # Moderate Risk
        else:
            return 1  # Low Risk
            
    elif attribute == 'gender':  # Gender
        # Assume female as slightly higher risk due to health-related factors, 
        # while balancing with the general risk level applicable to both
        if value == 'Female':
            return 2  # Moderate Risk
        elif value == 'Male':
            return 1  # Low Risk
        else:
            return 1  # Default Low Risk if no specific gender data available
    
    elif attribute == 'classification':  # Urban/Rural Classification
        if value == 'Urban':
            return 2  # Moderate Risk for urban areas (higher potential for spread)
        elif value == 'Rural':
            return 1  # Lower Risk for rural areas (less density)

    return 0  # Default case if no match found

# Example function to calculate final risk level
def calculate_final_risk(prediction):
    attributes = ['Predicted_Cases', 't2m_max', 'prectotcorr', 'ws2m_max', 
                  'rh2m', 'population_2020', 'average_household_size', 'agegroup',
                  'gender', 'classification']
    total_score = 0
    for attr in attributes:
        total_score += get_risk_score(attr, prediction[attr])
    
    # Get average risk score
    average_score = total_score / len(attributes)
    
    # Map average score to final risk level
    if average_score >= 3.5:
        return "Severe/Critical Risk"
    elif average_score >= 2.5:
        return "High Risk"
    elif average_score >= 1.5:
        return "Moderate Risk"
    elif average_score >= 0.5:
        return "Low Risk"
    else:
        return "No Risk"

model_path = 'C:/Users/Joseph Collantes/Desktop/THESIS/rfr_final_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        files = {
            'file1': 'weather',
            'file2': 'dengue',
            'file3': 'demographic',
            'file4': 'agegroup',
            'file5': 'gendergroup',
            'file6': 'surveydata'
        }
        dfs = {}
        expected_columns = {
            'weather': ['RH2M', 'T2M_MAX', 'WD2M', 'QV2M'],
            'dengue': ['DateOfEntry', 'Muncity', 'Barangay', 'LabTest', 'Year',
                        'MorbidityWeek','MuncityOfDRU', 'MorbidityMonth',
                        'AdmitToEntry', 'OnsetToAdmit', 'LabRes', 'Sex', 'Admitted',
                        'Type', 'ClinClass', 'CaseClassification', 'Outcome'],
            'demographic': ['Barangay'],
            'agegroup': ['muncity'],
            'gendergroup': ['muncity'],
            'surveydata': []
        }

        for file_key, file_type in files.items():
            if file_key in request.files:
                file = request.files[file_key]
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)

                    public_url = upload_to_firebase(file, filename, file_type)

                    try:
                        df = read_file(file_path)
                    except ValueError as e:
                        flash(f'File processing error: {e}', 'error')
                        return redirect(url_for('predict'))

                    if not all(col in df.columns for col in expected_columns[file_type]):
                        flash(f'File "{filename}" is missing required columns for {file_type}.', 'error')
                        return redirect(url_for('predict'))
                    
                    dfs[file_key] = df

        if len(dfs) != len(files):
            flash('Please upload all required files.', 'error')
            return redirect(url_for('predict'))

        weather_df = dfs['file1']
        dengue_df = dfs['file2']
        demographic_df = dfs['file3']
        agegroup_df = dfs['file4']
        gendergroup_df = dfs['file5']
        survey_df = dfs['file6']
        
        if 'Municipality' not in weather_df.columns and 'municipality' not in weather_df.columns:
            flash("Error: The weather_data does not contain a 'Municipality' or 'municipality' column.", "error")
            return redirect(url_for('predict'))
        
        if 'YEAR' in weather_df.columns and 'DOY' in weather_df.columns:
            weather_df['datemonthyear'] = weather_df.apply(lambda row: doy_to_date(row['YEAR'], row['DOY']), axis=1)
            
            weather_df = weather_df.drop(columns=['YEAR', 'DOY'])
            
            weather_df['datemonthyear'] = pd.to_datetime(weather_df['datemonthyear'], format='%d/%m/%Y')

        weather_df.columns = [col.lower() for col in weather_df.columns]

        string_cols = weather_df.select_dtypes(include='object').columns
        weather_df[string_cols] = weather_df[string_cols].apply(lambda x: x.str.lower())
        
        dengue_df = convert_date_columns(dengue_df)

        dengue_df.columns = dengue_df.columns.str.lower()
        
        age_columns = ['ageyears', 'agemons', 'agedays']

        if all(col in dengue_df.columns for col in age_columns):
            dengue_df['age'] = (dengue_df['ageyears'] +
                                dengue_df['agemons'] / 12 +
                                dengue_df['agedays'] / 365)

            dengue_df.drop(columns=age_columns, inplace=True)
        
        columns_to_remove = ['region', 'province', 'provofdru', 'datedied', 'icd10code',
                            'sentinelsite', 'deleterecord', 'recstatus', 'unnamed: 32',
                            'ilhz', 'district', 'typehospitalclinic', 'sent', 'ip', 'ipgroup']

        dengue_df = dengue_df.drop(columns=[col for col in columns_to_remove if col in dengue_df.columns])

        dengue_df = dengue_df.dropna(axis=1, how='all')
        
        dengue_df['barangay'] = dengue_df.groupby('muncity')['barangay'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))

        dengue_df = dengue_df.ffill()
        dengue_df = dengue_df.bfill()
        
        # Group by 'DOnset', 'Muncity', and 'Barangay', counting the number of occurrences
        combined_df = dengue_df.groupby(['donset', 'muncity', 'barangay']).size().reset_index()

        # Rename the last column to 'Cases'
        combined_df.columns = ['donset', 'muncity', 'barangay', 'cases']
        
        # Clean 'barangay' in demographic_df
        demographic_df.columns = demographic_df.columns.str.lower()
        demographic_df['barangay'] = demographic_df['barangay'].str.lower().replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        
        # Transform column names to lowercase
        agegroup_df.columns = agegroup_df.columns.str.lower()

        # Transform values in the 'Municipality' and 'AgeGroup' columns to lowercase
        agegroup_df['muncity'] = agegroup_df['muncity'].str.lower()
        agegroup_df['agegroup'] = agegroup_df['agegroup'].str.lower()

        # Remove the 'years' text from the 'agegroup' column
        agegroup_df['agegroup'] = agegroup_df['agegroup'].str.replace('years', '').str.strip()
                
        # Transform column names to lowercase
        gendergroup_df.columns = gendergroup_df.columns.str.lower()

        # Transform specific string columns to lowercase
        gendergroup_df['gender'] = gendergroup_df['gender'].str.lower()
        
        # Transform column names to lowercase
        survey_df.columns = survey_df.columns.str.lower()

        # Convert all string values to lowercase in each column
        survey_df = survey_df.apply(lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x))
        
        # Drop the gender column
        survey_df = survey_df.drop(columns=['gender'])
        
        # Merge combined_df with dengue_df on 'DOnset', 'Muncity', and 'Barangay'
        merged_df = pd.merge(dengue_df, combined_df, on=['donset', 'muncity', 'barangay'], how='left')
        
        # Ensure 'DOnset' is in datetime format
        merged_df['donset'] = pd.to_datetime(merged_df['donset'], errors='coerce')

        # Determine the minimum and maximum dates
        min_date = merged_df['donset'].min()
        max_date = merged_df['donset'].max()

        # Generate a date range from min_date to max_date
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        # Create a DataFrame from the date range
        daily_dates_df = pd.DataFrame(date_range, columns=['datemonthyear'])
        daily_dates_df['datemonthyear'] = pd.to_datetime(daily_dates_df['datemonthyear'], format='%d/%m/%Y')
        
        # Merge the daily dates DataFrame with merged_df on 'donset'
        merged_daily_df = pd.merge(daily_dates_df, merged_df, how='left', left_on='datemonthyear', right_on='donset')

        # Drop datetime columns except for 'datemonthyear'
        datetime_cols = merged_daily_df.select_dtypes(include=['datetime']).columns.tolist()
        columns_to_drop = [col for col in datetime_cols if col != 'datemonthyear']
        merged_daily_df = merged_daily_df.drop(columns=columns_to_drop)
        
        # Ensure 'datemonthyear' column is of datetime type
        merged_daily_df['datemonthyear'] = pd.to_datetime(merged_daily_df['datemonthyear'])
        weather_df['datemonthyear'] = pd.to_datetime(weather_df['datemonthyear'])

        # Merge on 'datemonthyear', keeping rows with matching muncity and municipality values
        merged_data = pd.merge(merged_daily_df, weather_df, on='datemonthyear', how='inner')

        # Filter rows where 'muncity' in merged_daily_df equals 'municipality' in weather_df
        filtered_data = merged_data[merged_data['muncity'] == merged_data['municipality']].copy()
        
        # Drop the 'municipality' column
        filtered_data.drop(columns=['municipality'], inplace=True)  # This might need .copy()
        filtered_data.loc[:, 'datemonthyear'] = pd.to_datetime(filtered_data['datemonthyear'], errors='coerce')

        # Determine the minimum and maximum dates
        min_date = filtered_data['datemonthyear'].min()
        max_date = filtered_data['datemonthyear'].max()

        # Generate a date range from min_date to max_date
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        # Create a DataFrame from the date range
        daily_dates = pd.DataFrame(date_range, columns=['datemonthyear'])
        daily_dates['datemonthyear'] = pd.to_datetime(daily_dates['datemonthyear'], format='%d/%m/%Y', dayfirst=True)

        # Merge the daily dates DataFrame with filtered_data
        final_merge = pd.merge(daily_dates, filtered_data, how='left', left_on='datemonthyear', right_on='datemonthyear')

        # Drop unnecessary datetime columns
        datetime_cols = final_merge.select_dtypes(include=['datetime']).columns.tolist()
        columns_to_drop = [col for col in datetime_cols if col != 'datemonthyear']
        final_merge = final_merge.drop(columns=columns_to_drop)

        final_merge['monthyear'] = final_merge['datemonthyear'].dt.to_period('M')
        
        # Clean 'barangay' in final_merge
        final_merge['barangay'] = final_merge['barangay'].str.replace('pob', '', case=False, regex=True).str.strip()
        
        # Merge the datasets on both 'muncity' and 'barangay' columns
        merged_demographic = pd.merge(final_merge, demographic_df, on=['muncity', 'barangay'], how='left')        
        
        # Fill missing values with 0
        cleaned_merged_demographic = merged_demographic.fillna(0)
        
        # Transform the 'age' column into whole numbers
        cleaned_merged_demographic['age'] = cleaned_merged_demographic['age'].astype(int)
        
        # Merge the two datasets based on 'municipality' from age_group and 'muncity' from cleaned_merge
        merged_agegroup = pd.merge(cleaned_merged_demographic, agegroup_df, on='muncity', how='left')
        
        # Apply the function to create a new column 'age_range' in merged_df
        merged_agegroup['age_range'] = merged_agegroup['age'].apply(map_age_to_agegroup)
        
        # Filter the DataFrame: keep rows where 'age_range' equals 'agegroup' or cases equals 0
        filtered_agegroup = merged_agegroup[(merged_agegroup['age_range'] == merged_agegroup['agegroup']) | (merged_agegroup['cases'] == 0)].copy()
        
        # Fill NaN values with 0 in the filtered DataFrame
        filtered_agegroup.fillna(0, inplace=True)
        
        # Merge the DataFrames on the 'muncity' column without dropping rows with 0 cases
        merged_gendergroup = pd.merge(filtered_agegroup, gendergroup_df, on='muncity', how='left')
        
        # Filter rows: Keep rows where 'sex' equals 'gender' or 'cases' is 0
        filtered_merged_data = merged_gendergroup[
            (merged_gendergroup['sex'].str.lower() == merged_gendergroup['gender']) | (merged_gendergroup['cases'] == 0)
        ].copy()  # Create a copy of the filtered DataFrame

        # Fill NaN values with 0
        filtered_merged_data.fillna(0, inplace=True)

        # Drop the 'sex' column
        filtered_merged_data.drop(columns=['sex'], inplace=True)
        
        print(filtered_merged_data.info())

        # Create a mask to find the rows where the cases column is 0
        target_length = 2715

        # Randomly sample rows from survey_data with replacement to reach target_length
        survey_data_repeated = survey_df.sample(n=target_length, replace=True, random_state=42).reset_index(drop=True)
        
        
        # Ensure the indices of survey_data_repeated and filtered_merged_data match
        survey_data_repeated = survey_data_repeated.reset_index(drop=True)
        filtered_merged_data = filtered_merged_data.reset_index(drop=True)
        
        mask = filtered_merged_data['cases'] == 0

        # Set the values in survey_data_repeated to 0 for the rows where the mask is True
        survey_data_repeated.loc[mask] = 0

        # Combine the survey_data_repeated with filtered_merged_data on the left side
        training_data = pd.concat([survey_data_repeated, filtered_merged_data], axis=1)
        training_data.fillna(0, inplace=True)

        # Identify categorical columns
        categorical_cols = training_data.select_dtypes(include=['object']).columns

        # Initialize LabelEncoders and create a dictionary to save them
        label_encoders = {}

        for col in categorical_cols:
            training_data[col] = training_data[col].astype(str)
            label_encoder = LabelEncoder()
            training_data[col] = label_encoder.fit_transform(training_data[col])
            # Save the label encoder for reverse mapping
            label_encoders[col] = label_encoder

        # Update features for prediction
        features = ['t2m_max', 'wd2m', 'rh2m', 'qv2m', 'population (2020)', 'caseclassification',
                    'gender', 'agegroup', 'barangay', 'outcome', 'morbiditymonth', 'muncity', 'clinclass',
                    'annual regular income', 'average household size', 'living conditions (ventilation, screens on windows, closed environment)',
                    'how many times do you clean the house in a week? inside:', 'source of water supply']

        X_new = training_data[features]

        # Perform predictions
        training_data['Predicted_Cases'] = model.predict(X_new)

        # Ensure 'datemonthyear' is a datetime before adding offset
        training_data['datemonthyear'] = pd.to_datetime(training_data['datemonthyear'])
        training_data['datemonthyear'] = training_data['datemonthyear'] + pd.DateOffset(years=1)

        # Filter for next year's predictions
        next_year = training_data['datemonthyear'].dt.year.max()
        predictions_next_year = training_data[training_data['datemonthyear'].dt.year == next_year]

        # Group predictions by day
        daily_predictions = predictions_next_year.groupby(['datemonthyear', 'muncity', 'barangay', 't2m_max', 'prectotcorr',
                                                           'ws2m_max', 'rh2m', 'population (2020)', 'average household size',
                                                           'agegroup', 'gender', 'classification','source of water supply',
                                                           'living conditions (ventilation, screens on windows, closed environment)',
                                                           'mosquito control measures', 'access to local health facilities',
                                                           'time spent outdoors (especially at dusk and dawn)']).agg({
        'Predicted_Cases': 'sum'
        }).reset_index()
                                                           
        print(daily_predictions['Predicted_Cases'].head(60))

        # Filter out rows where 'barangay' or 'muncity' is NaN or 0
        daily_predictions = daily_predictions[daily_predictions['barangay'].notna() & (daily_predictions['barangay'] != 0)]
        daily_predictions = daily_predictions[daily_predictions['muncity'].notna() & (daily_predictions['muncity'] != 0)]

        # Reverse encoding for muncity and barangay using the saved label encoders
        daily_predictions['muncity'] = label_encoders['muncity'].inverse_transform(daily_predictions['muncity'])
        daily_predictions['barangay'] = label_encoders['barangay'].inverse_transform(daily_predictions['barangay'])
        daily_predictions['gender'] = label_encoders['gender'].inverse_transform(daily_predictions['gender'])
        daily_predictions['agegroup'] = label_encoders['agegroup'].inverse_transform(daily_predictions['agegroup'])
        daily_predictions['classification'] = label_encoders['classification'].inverse_transform(daily_predictions['classification'])
        daily_predictions['source of water supply'] = label_encoders['source of water supply'].inverse_transform(daily_predictions['source of water supply'])
        daily_predictions['living conditions (ventilation, screens on windows, closed environment)'] = label_encoders['living conditions (ventilation, screens on windows, closed environment)'].inverse_transform(daily_predictions['living conditions (ventilation, screens on windows, closed environment)'])
        daily_predictions['mosquito control measures'] = label_encoders['mosquito control measures'].inverse_transform(daily_predictions['mosquito control measures'])
        daily_predictions['access to local health facilities'] = label_encoders['access to local health facilities'].inverse_transform(daily_predictions['access to local health facilities'])
        daily_predictions['time spent outdoors (especially at dusk and dawn)'] = label_encoders['time spent outdoors (especially at dusk and dawn)'].inverse_transform(daily_predictions['time spent outdoors (especially at dusk and dawn)'])
        
        # Round the predicted cases to whole numbers
        daily_predictions['Predicted_Cases'] = daily_predictions['Predicted_Cases'].fillna(0).round().astype(int)
        print(daily_predictions['Predicted_Cases'].head(60))

        
        # # Filter out rows where 'Predicted_Cases' is 0 after rounding
        # daily_predictions = daily_predictions[daily_predictions['Predicted_Cases'] != 0]
        
        # print(daily_predictions['Predicted_Cases'].head(60))

        # Format the 'datemonthyear' column to display as 'Month Day, Year'
        daily_predictions['datemonthyear'] = daily_predictions['datemonthyear'].dt.strftime('%B %d, %Y')
        
        # Convert the results to a dictionary for rendering
        predictions = daily_predictions.to_dict(orient='records')
        
        for prediction in predictions:
            prediction['population_2020'] = prediction.pop('population (2020)')
            prediction['average_household_size'] = prediction.pop('average household size')
            prediction['source_of_water_supply'] = prediction.pop('source of water supply')
            prediction['living_conditions'] = prediction.pop('living conditions (ventilation, screens on windows, closed environment)')
            prediction['mosquito_control_measures'] = prediction.pop('mosquito control measures')
            prediction['access_to_local_health_facilities'] = prediction.pop('access to local health facilities')
            prediction['time_spent_outdoors'] = prediction.pop('time spent outdoors (especially at dusk and dawn)')
            prediction['Risk_Level'] = calculate_final_risk(prediction)

        # Render the prediction results
        return render_template('predict.html', predictions=predictions)

    return render_template('predict.html')	

@app.route('/contact_numbers', methods=['GET'])
def get_contact_numbers():
    # Initialize Firestore client
    db = firestore.client()

    # Reference to the contact numbers document
    contact_numbers_ref = db.collection('admin_acc').document('contact_numbers')
    contact_numbers_doc = contact_numbers_ref.get()

    if contact_numbers_doc.exists:
        contact_numbers = contact_numbers_doc.to_dict()
        return jsonify(contact_numbers)
    else:
        return jsonify({'error': 'Contact numbers not found'}), 404
    
@app.route('/update-contact-numbers', methods=['POST'])
def update_contact_numbers():
    data = request.json
    print('Received data:', data)  # Debugging output
    try:
        # Initialize Firestore client
        db = firestore.client()

        # Reference to the contact numbers document
        contact_numbers_ref = db.collection('admin_acc').document('contact_numbers')

        # Get the existing document data
        existing_data = contact_numbers_ref.get().to_dict() or {}

        # Update the document with the new contact numbers
        existing_data.update(data)
        contact_numbers_ref.set(existing_data)
        return jsonify({'success': True}), 200
    except Exception as e:
        print('Error updating contact numbers:', e)  # Debugging output
        return jsonify({'success': False}), 500

@app.after_request
def after_request(response):
    if request.endpoint in ['user_page', 'admin_page']:
        set_no_cache(response)
    return response

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            # Check in admin_acc collection first
            admin_ref = db.collection('admin_acc').where('username', '==', username).stream()
            admin = next(admin_ref, None)

            if admin:
                admin_data = admin.to_dict()
                if check_password_hash(admin_data['password'], password):  # Hash comparison
                    session['admin'] = username  # Set session for admin
                    return redirect(url_for('admin_page'))

            # Check in user_acc collection if not an admin
            user_ref = db.collection('user_acc').where('email', '==', username).stream()
            user = next(user_ref, None)

            if user:
                user_data = user.to_dict()
                if check_password_hash(user_data['password'], password):  # Hash comparison
                    if user_data['status'] == 'approved':
                        session['user'] = username  # Set session for user
                        return redirect(url_for('user_page'))
                    else:
                        flash('Your account is not approved. Please contact support.')
                        return redirect(url_for('login'))

            flash('Invalid username or password')
            return redirect(url_for('login'))

        except Exception as e:
            print(f"Error: {e}")
            flash('Database connection failed')
            return redirect(url_for('login'))

    error = request.args.get('error')
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('admin', None)
    return redirect(url_for('login'))

@app.route('/admin')
def admin_page():
    if 'admin' not in session:
        return redirect(url_for('login'))

    selected_municipality = request.args.get('municipality')

    municipalities = ['Famy', 'Kalayaan', 'Mabitac', 'Paete', 'Pakil', 'Pangil', 'Santa Maria', 'Siniloan']
    admin_plots = {}
    admin_yearly_plots = {}

    # Process data for each municipality
    for municipality in municipalities:
        (_, _, _, _, _, municipality_plots, municipality_yearly_plots, _, _, _, _) = process_data(municipality)
        admin_plots[municipality] = municipality_plots.get(municipality, "")
        admin_yearly_plots.update(municipality_yearly_plots)

    # Process data for the selected municipality
    (month_distribution_html, age_histogram_html, gender_bar_html, cases_per_municipality_html,
     cases_per_municipality_data, _, monthly_admission_plots, monthly_trend_html, yearly_distribution_html,
     total_cases, cases_per_barangay) = process_data(selected_municipality)

    return render_template('admin.html',
                           month_distribution_html=month_distribution_html,
                           age_histogram_html=age_histogram_html,
                           gender_bar_html=gender_bar_html,
                           cases_per_municipality_html=cases_per_municipality_html,
                           plots=admin_plots,
                           selected_municipality=selected_municipality,
                           cases_per_municipality_data=cases_per_municipality_data,
                           monthly_admission_plots=monthly_admission_plots,
                           monthly_trend_html=monthly_trend_html,
                           yearly_distribution_html=yearly_distribution_html,
                           total_cases=total_cases,
                           cases_per_barangay=cases_per_barangay)

@app.route('/update_plot')
def update_plot():
    municipality = request.args.get('municipality')
    
    if municipality:
        _, _, _, _, cases_per_municipality_html, plots, _, _ = process_data(municipality)
        return jsonify({
            'casesPerMunicipality': cases_per_municipality_html,
            'barangayPlot': plots.get(municipality, "")
        })
    else:
        return jsonify({'error': 'No municipality selected'})
    
@app.route('/repository', methods=['GET'])
def repository():
    blobs = bucket.list_blobs()
    
    folders = {}
    expiration = datetime.utcnow() + timedelta(minutes=10)  
    
    for blob in blobs:
        folder_name = blob.name.split('/')[0]  
        file_info = {
            'name': blob.name,
            'url': blob.generate_signed_url(expiration=expiration)  
        }
        if folder_name not in folders:
            folders[folder_name] = []
        folders[folder_name].append(file_info)
    
    folder_list = [{'name': name, 'files': files} for name, files in folders.items()]
    
    return render_template('repository.html', folders=folder_list)

@app.route('/registration')
def registration():
    return render_template('registration.html')

@app.route('/feedbacks')
def feedbacks():
    try:
        feedback_ref = db.collection('feedbacks')
        feedbacks = feedback_ref.stream()

        feedback_list = []
        for feedback in feedbacks:
            data = feedback.to_dict()
            feedback_list.append({
                'id': feedback.id, 
                'Date and Time': data.get('Date and Time'),
                'Fullname': data.get('Fullname'),
                'Email': data.get('Email'),
                'Feedback': data.get('Feedback'),
                'Action': data.get('Action')
            })
        
        total_feedbacks = len(feedback_list)

        return render_template('feedbacks.html', feedbacks=feedback_list, total_feedbacks=total_feedbacks)
    except Exception as e:
        print(f"Error fetching feedbacks: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    fullname = request.form.get('fullname')
    email = request.form.get('email')
    feedback = request.form.get('feedback')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if not fullname or not email or not feedback:
        flash("All fields are required.", "error")
        return redirect(request.referrer)  # Redirect back to the previous page

    try:
        feedback_ref = db.collection('feedbacks')
        feedback_ref.add({
            'Fullname': fullname,
            'Email': email,
            'Feedback': feedback,
            'Date and Time': timestamp,
            'Action': 'None'
        })

        flash("Feedback submitted successfully!", "success")
        return redirect(request.referrer)  # Redirect back to the previous page
    except Exception as e:
        print(f"Error submitting feedback: {e}")
        flash("An error occurred while submitting feedback. Please try again.", "error")
        return redirect(request.referrer)  # Redirect back to the previous page
    
@app.route('/update_action', methods=['POST'])
def update_action():
    data = request.get_json()
    feedback_id = data.get('id') 
    new_action = data.get('action')

    try:
        feedback_ref = db.collection('feedbacks').document(feedback_id)
        feedback_ref.update({
            'Action': new_action
        })
        return jsonify({"message": "Action updated successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/messages')
def messages():
    try:
        message_ref = db.collection('messages')
        messages = message_ref.stream()

        message_list = []
        for message in messages:
            data = message.to_dict()
            message_list.append({
                'id': message.id, 
                'date_and_time': data.get('Date and Time'),
                'fullname': data.get('Fullname'),
                'email': data.get('Email'),
                'contact_number': data.get('Contact Number'),
                'message': data.get('Message'),
                'action': data.get('Action')
            })
        
        total_messages = len(message_list)

        return render_template('messages.html', messages=message_list, total_messages=total_messages)
    except Exception as e:
        print(f"Error fetching messages: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/submit_message', methods=['POST'])
def submit_message():
    fullname = request.form.get('fullname')
    contactnumber = request.form.get('contactnumber')
    email = request.form.get('email')
    message = request.form.get('message')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if not fullname or not contactnumber or not email or not message:
        flash("All fields are required.", "error")
        return redirect(request.referrer)  # Redirect back to the previous page

    try:
        message_ref = db.collection('messages')
        message_ref.add({
            'Fullname': fullname,
            'Contact Number': contactnumber,
            'Email': email,
            'Message': message,
            'Date and Time': timestamp,
            'Action': 'None'
        })

        flash("Message submitted successfully!", "success")
        return redirect(request.referrer)  # Redirect back to the previous page
    except Exception as e:
        print(f"Error submitting message: {e}")
        flash("An error occurred while submitting your message. Please try again.", "error")
        return redirect(request.referrer)  # Redirect back to the previous page
    
@app.route('/message_update_action', methods=['POST'])
def message_update_action():
    data = request.get_json()
    message_id = data.get('id') 
    new_action = data.get('action')

    try:
        message_ref = db.collection('messages').document(message_id)
        message_ref.update({
            'Action': new_action
        })
        return jsonify({"message": "Action updated successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/forgotpass', methods=['GET', 'POST'])
def forgotpass():
    if request.method == 'POST':
        phone = request.form.get('phone').strip()
        print(f'Attempting to send OTP to phone: {phone}')
        
        try:
            # Check if phone number exists in the admin_acc collection
            admin_users = db.collection('admin_acc').where('phone', '==', phone).get()

            if len(admin_users) == 0:
                return render_template('forgotpass.html', error='Phone number not found.')

            # Generate OTP
            otp = random.randint(100000, 999999)
            session['otp'] = otp
            session['phone'] = phone
            print(f"Generated OTP: {otp}")  # Debugging line

            # Send OTP using Semaphore
            try:
                message = f"Your One Time Password is: {otp}. Please use it within 5 minutes."
                data = {
                    "apikey": "f6fd867099d3b992b51e3eb7a0bd8dc8",
                    "number": phone,
                    "message": message
                }
                response = requests.post("https://api.semaphore.co/api/v4/otp", data=data)
                
                if response.status_code != 200:
                    print(f'Error: {response.status_code}, Response: {response.text}')
                    raise Exception('Failed to send OTP via SMS.')

            except Exception as e:
                print(f'Error sending OTP: {e}')
                return render_template('forgotpass.html', error='Failed to send OTP. Please try again.')

            return redirect(url_for('verify_otp'))
        
        except Exception as e:
            print(f'General error: {e}')
            return render_template('forgotpass.html', error='There was an error processing your request.')

    return render_template('forgotpass.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if 'otp' not in session:  # Check if OTP was generated
        flash('You must request a password reset first.', 'error')
        return redirect(url_for('forgotpass'))

    if request.method == 'POST':
        entered_otp = request.form.get('otp')

        # Validate that the entered OTP is a number
        try:
            entered_otp = int(entered_otp)
        except ValueError:
            return render_template('verify_otp.html', error='Invalid input. Please enter a numeric OTP.')

        # Compare the entered OTP with the OTP in the session
        session_otp = session.get('otp')
        print(f"Entered OTP: {entered_otp}, Session OTP: {session_otp}")  # Debugging line

        if entered_otp == session_otp:
            return redirect(url_for('resetpass'))  # Redirect to password reset page
        else:
            return render_template('verify_otp.html', error='Invalid OTP. Please try again.')

    return render_template('verify_otp.html')


@app.route('/resetpass', methods=['GET', 'POST'])
def resetpass():
    if 'otp' not in session:  # Check if OTP was generated
        flash('You must request a password reset first.', 'error')
        return redirect(url_for('forgotpass'))

    if request.method == 'POST':
        new_password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Check if passwords match
        if new_password != confirm_password:
            return render_template('resetpass.html', error='Passwords do not match.')

        try:
            # Get the admin document directly (assuming you're looking for the first document)
            admin_ref = db.collection('admin_acc').limit(1).get()

            if not admin_ref:
                return render_template('resetpass.html', error='Admin account not found.')

            # Hash the new password
            hashed_password = generate_password_hash(new_password)

            # Update password in the admin_acc collection
            admin_doc = admin_ref[0].reference
            admin_doc.update({'password': hashed_password})

            # Clear OTP from session after successful password reset
            session.pop('otp', None)
            session.pop('phone', None)

            return render_template('resetpass.html', success='Your password has been successfully reset.', redirect=True)
        except Exception as e:
            print(f'Error resetting admin password: {e}')
            return render_template('resetpass.html', error='Error resetting password. Please try again.')

    return render_template('resetpass.html')

@app.route('/')
def home():
    error = request.args.get('error')
    
    selected_municipality = request.args.get('municipality')

    municipalities = ['Famy', 'Kalayaan', 'Mabitac', 'Paete', 'Pakil', 'Pangil', 'Santa Maria', 'Siniloan']
    user_plots = {}
    user_yearly_plots = {}

    for municipality in municipalities:
        (_, _, _, _, _, municipality_plots, municipality_yearly_plots, _, _, _, _) = process_data(municipality)
        user_plots[municipality] = municipality_plots.get(municipality, "")
        user_yearly_plots.update(municipality_yearly_plots)

    (month_distribution_html, age_histogram_html, gender_bar_html, cases_per_municipality_html,
     cases_per_municipality_data, _, monthly_admission_plots, monthly_trend_html, yearly_distribution_html,
     total_cases, cases_per_barangay) = process_data(selected_municipality)

    return render_template('home.html', error=error,
                           month_distribution_html=month_distribution_html,
                           age_histogram_html=age_histogram_html,
                           gender_bar_html=gender_bar_html,
                           cases_per_municipality_html=cases_per_municipality_html,
                           plots=user_plots,
                           selected_municipality=selected_municipality,
                           cases_per_municipality_data=cases_per_municipality_data,
                           monthly_admission_plots=monthly_admission_plots,
                           monthly_trend_html=monthly_trend_html,
                           yearly_distribution_html=yearly_distribution_html,
                           total_cases=total_cases,
                           cases_per_barangay=cases_per_barangay)


@app.route('/send-sms', methods=['POST'])
def send_sms():
    try:
        data = request.json
        url = 'https://api.semaphore.co/api/v4/messages'

        numbers = data.get('numbers', [])  
        message = data.get('message', '')

        valid_numbers = []
        for number in numbers:
            if re.match(r'^(?:\+63|0)[0-9]{10}$', number.strip()):
                if number.startswith('0'):
                    number = '+63' + number[1:]  
                valid_numbers.append(number.strip())

        if not valid_numbers or not message:
            return jsonify({"error": "Valid numbers and message are required."}), 400

        payload = {
            'apikey': 'f6fd867099d3b992b51e3eb7a0bd8dc8',
            'number': ','.join(valid_numbers),  
            'message': message,
            'sendername': 'SEMAPHORE'  
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()  
        return jsonify(response.json()), response.status_code
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
