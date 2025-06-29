import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data):
    """
    Perform Exploratory Data Analysis (EDA) on the given DataFrame.
    
    Parameters:
    data (DataFrame): The preprocessed DataFrame on which EDA is to be performed.
    """    
    print("First few rows of the data:")
    print(data.head())

    for col in data.columns:
       if col is ['BMI']:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[col], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    # Summary statistics for features
    print("\nSummary Statistics for Features:")
    print(data.describe())
    print("\n")

    # Heatmap for feature correlations
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Heatmap of Feature Correlations")
    plt.show()

    # Correlation with the target 'Heart_Disease_Status'
    plt.figure(figsize=(10, 8))
    target_correlation = data.corr()['Heart_Disease_Status'].sort_values(ascending=False)
    sns.barplot(x=target_correlation.values, y=target_correlation.index, palette="viridis")
    plt.title("Correlation of Features with Heart_Disease_Status")
    plt.xlabel("Correlation Coefficient")
    plt.show()
    print(target_correlation)

    # Plot histograms for continuous features
    for col in data.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[col], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

# Example usage with your DataFrame
df = pd.read_csv('smote.csv')  # Load your preprocessed dataset
perform_eda(df)  # Call the EDA function on your DataFrame
