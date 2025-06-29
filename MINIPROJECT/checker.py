import os
import pandas as pd
import random
import numpy as np

# Load the dataset
df = pd.read_csv(f'smote.csv')
print(len(df.columns))
# Display unique values for each column
for column in df.columns:
    print(f"Unique values for {column}:")
    print(df[column].unique())
    print("\n")
