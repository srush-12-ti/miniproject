import pandas as pd

def preprocess_data(data):
    """
    Preprocess the DataFrame to make changes to BMI, Mental Health, Physical Health, 
    and combine Fruits and Veggies into a single feature.
    
    Parameters:
    data (DataFrame): The DataFrame to be preprocessed.
    
    Returns:
    DataFrame: The preprocessed DataFrame.
    """
    # 1️⃣ **Convert BMI to BMI classes**
    data['BMI'] = pd.cut(
        data['BMI'], 
        bins=[-float('inf'), 18.5, 25, 30, float('inf')],  # Classify BMI
        labels=[0, 1, 2, 3]  # Corresponding classes: Underweight, Healthy, Overweight, Obesity
    ).astype(int)

    # 2️⃣ **Bin Mental Health into 6 bins (0-5, 5-10, ..., 25-30)**
    data['MentHlth'] = pd.cut(
        data['MentHlth'], 
        bins=[-float('inf'), 5, 10, 15, 20, 25, 30],  
        labels=[0, 1, 2, 3, 4, 5]  # Corresponding bins
    ).astype(int)

    # 3️⃣ **Bin Physical Health into 6 bins (0-5, 5-10, ..., 25-30)**
    data['PhysHlth'] = pd.cut(
        data['PhysHlth'], 
        bins=[-float('inf'), 5, 10, 15, 20, 25, 30], 
        labels=[0, 1, 2, 3, 4, 5]  # Corresponding bins
    ).astype(int)

    # 4️⃣ **Combine Fruits and Veggies into a single feature**
    # If both Fruits and Veggies are 1, then Fruits_Veggies = 1
    # Otherwise, Fruits_Veggies = 0
    data['Fruits_Veggies'] = (data['Fruits'] | data['Veggies'])  # Binary AND operation

    # Drop the original 'Fruits' and 'Veggies' columns if no longer needed
    data = data.drop(columns=['Fruits', 'Veggies'], errors='ignore')
    
    return data


df = pd.read_csv('finaldataset.csv')  # Load your preprocessed dataset
df = preprocess_data(df)  # Apply the preprocessing function
df.to_csv('binned.csv', index=False)  # Save the updated DataFrame
