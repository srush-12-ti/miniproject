import pandas as pd

# Load the dataset
df = pd.read_csv('smote.csv')

# Convert values of Heart_Disease_Status from 4 to 1
df['Heart_Disease_Status'] = df['Heart_Disease_Status'].replace(4, 0)

# Save the modified dataset back to the same file
df.to_csv('smote.csv', index=False)

print("Conversion complete. Values of 4 in 'Heart_Disease_Status' have been changed to 1.")