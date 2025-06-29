import os
import pandas as pd
import random
import numpy as np

# Set random seed for reproducibility
random.seed(1)

# Load the dataset for the required year
year = '2015'
brfss_2015_dataset = pd.read_csv(f'{year}.csv')

# Select specific columns (with required adjustments, excluding 'BPHIGH4' and 'HighBP')
columns_to_select = [
    'TOLDHI2', '_BMI5', 'CVDSTRK3', 'DIABETE3', '_TOTINDA', 
    '_FRTLT1', '_VEGLT1', '_RFDRHV5', 'GENHLTH', 'MENTHLTH', 'PHYSHLTH', 
    'DIFFWALK', 'SEX', 'EDUCA', 'INCOME2', 
    '_RFSMOK3', '_INCOMG', '_AGE_G', 'BPMEDS', 'CVDCRHD4', 'CVDINFR4', 'CVDSTRK3'  # New variables added
]

# Drop unnecessary columns
brfss_df_selected = brfss_2015_dataset[columns_to_select]

# Remove duplicate columns
brfss_df_selected = brfss_df_selected.loc[:, ~brfss_df_selected.columns.duplicated()]

# Clean column names by stripping whitespace and removing special characters
brfss_df_selected.columns = brfss_df_selected.columns.str.strip().str.replace(' ', '')

# Drop missing values
brfss_df_selected = brfss_df_selected.replace([77,99,'7', '9', 'BLANK', 'Don’t know/Not sure', 'Refused', 'Missing'], pd.NA)
print(brfss_df_selected["TOLDHI2"].unique())
brfss_df_selected = brfss_df_selected.dropna()
print(brfss_df_selected["TOLDHI2"].unique())

# Convert specific variables to binary or categorical format
brfss_df_selected['TOLDHI2'] = brfss_df_selected['TOLDHI2'].replace({2: 0, 1: 1})  # High cholesterol: 0 = No, 1 = Yes
brfss_df_selected['_RFSMOK3'] = brfss_df_selected['_RFSMOK3'].replace({1: 0, 2: 1})  # Current smoker: 0 = No, 1 = Yes
brfss_df_selected['BPMEDS'] = brfss_df_selected['BPMEDS'].replace({1: 1, 2: 0})  # Taking BP meds: 0 = No, 1 = Yes
brfss_df_selected['MENTHLTH'] = brfss_df_selected['MENTHLTH'].replace({88: 0}) 
brfss_df_selected['PHYSHLTH'] = brfss_df_selected['PHYSHLTH'].replace({88: 0}) 

# Create the target variable 'Heart_Disease_Status' with a default of 4 (Healthy)
brfss_df_selected['Heart_Disease_Status'] = 4  # 4 = Healthy

# Update target status based on conditions
brfss_df_selected.loc[brfss_df_selected['CVDCRHD4'] == 1, 'Heart_Disease_Status'] = 1  # Coronary Heart Disease
brfss_df_selected.loc[brfss_df_selected['CVDINFR4'] == 1, 'Heart_Disease_Status'] = 2  # Myocardial Infarction
brfss_df_selected.loc[brfss_df_selected['CVDSTRK3'] == 1, 'Heart_Disease_Status'] = 3  # Stroke

# Drop original columns used to create the target
brfss_df_selected = brfss_df_selected.drop(columns=['CVDCRHD4', 'CVDINFR4', 'CVDSTRK3'], errors='ignore')

# Rename columns for clarity (excluding 'HighBP' and 'BPHIGH4')
rename_columns = {
    'TOLDHI2': 'HighChol', 
    '_BMI5': 'BMI', 
    'CVDSTRK3': 'Stroke', 
    'DIABETE3': 'Diabetes', 
    '_TOTINDA': 'PhysActivity', 
    '_FRTLT1': 'Fruits', 
    '_VEGLT1': 'Veggies', 
    '_RFDRHV5': 'HvyAlcoholConsump', 
    'GENHLTH': 'GenHlth', 
    'MENTHLTH': 'MentHlth', 
    'PHYSHLTH': 'PhysHlth', 
    'DIFFWALK': 'DiffWalk', 
    'SEX': 'Sex', 
    'EDUCA': 'Education', 
    'INCOME2': 'Income', 
    '_RFSMOK3': 'Current_Smoker', 
    '_INCOMG': 'Income_Category', 
    '_AGE_G': 'Age_Group', 
    'BPMEDS': 'On_BP_Medication'  
}

brfss_df_selected = brfss_df_selected.rename(columns=rename_columns)

# Save the preprocessed file
brfss_df_selected.to_csv(f'preprocessed{year}.csv', sep=',', index=False)

print(f"Preprocessed data saved to preprocessed{year}")

df = pd.read_csv(f'preprocessed{year}.csv')

# Convert all columns except the excluded ones
exclude_columns = ['BMI']
for column in df.columns:
    if column not in exclude_columns:
        df[column] = df[column].astype(int)  # Change to desired type (e.g., int, float)

# Clean specific columns (like HighBP and PhysActivity)
df['PhysActivity'] = df['PhysActivity'].replace({2: 0, 9: np.NaN})
df['HighChol'] = df['HighChol'].replace({7: np.NaN, 9: np.NaN})
df['BMI'] = df['BMI'] / 100  # Adjust BMI scale
df['Diabetes'] = df['Diabetes'].replace({2: 1, 3: 0, 4: 0, 7: np.NaN, 9: np.NaN})
df['Fruits'] = df['Fruits'].replace({2: 0, 9: np.NaN})
df['Veggies'] = df['Veggies'].replace({2: 0, 9: np.NaN})
df['HvyAlcoholConsump'] = df['HvyAlcoholConsump'].replace({1: 0, 2: 1, 9: np.NaN})
df['GenHlth'] = df['GenHlth'].replace({7: np.NaN, 9: np.NaN})
df['MentHlth'] = df['MentHlth'].replace({88: 0}) 
df['PhysHlth'] = df['PhysHlth'].replace({88: 0}) 
df['DiffWalk'] = df['DiffWalk'].replace({2: 0, 7: np.NaN, 9: np.NaN})
df['Sex'] = df['Sex'].replace({1: 0, 2: 1})
df['Education'] = df['Education'].replace({9: np.NaN})
df['Income'] = df['Income'].replace({77: np.NaN, 99: np.NaN})
df = df.drop(columns=['Income'])
df['Current_Smoker'] = df['Current_Smoker'].replace({9: np.NaN})
df['Income_Category'] = df['Income_Category'].replace({9: np.NaN})
#df['Age_Group'] = df['Age_Group'].replace({:})
df['On_BP_Medication'] = df['On_BP_Medication'].replace({7: np.NaN, 9: np.NaN})

# Drop any rows with NaN values
df = df.dropna()

# Convert all columns except the excluded ones
exclude_columns = ['BMI']
for column in df.columns:
    if column not in exclude_columns:
        df[column] = df[column].astype(int)  # Change to desired type (e.g., int, float)

print(df.head())

# Save the final processed data
output_path = f'preprocessed{year}.csv'
df.to_csv(output_path, sep=',', index=False)
print(f"Final preprocessed data saved to {output_path}")
