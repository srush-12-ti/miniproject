import joblib
import numpy as np

def get_user_input():
    """Collects user input for all the necessary features to make a prediction."""
    print("Please provide the following information for heart disease prediction:\n")
    
    try:
        HighChol = int(input("High Cholesterol (1 = Yes, 0 = No): "))
        BMI = float(input("BMI (Body Mass Index, e.g., 24.5): "))
        Diabetes = int(input("Diabetes (1 = Yes, 0 = No): "))
        PhysActivity = int(input("Physical Activity (1 = Yes, 0 = No): "))
        Fruits = int(input("Consumes Fruits Daily (1 = Yes, 0 = No): "))
        Veggies = int(input("Consumes Vegetables Daily (1 = Yes, 0 = No): "))
        HvyAlcoholConsump = int(input("Heavy Alcohol Consumption (1 = Yes, 0 = No): "))
        GenHlth = int(input("General Health (1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor): "))
        MentHlth = int(input("Days of Poor Mental Health (0-30): "))
        PhysHlth = int(input("Days of Poor Physical Health (0-30): "))
        DiffWalk = int(input("Difficulty Walking (1 = Yes, 0 = No): "))
        Sex = int(input("Sex (1 = Male, 0 = Female): "))
        Education = int(input("Education Level (1 = No Schooling, 2 = Elementary, 3 = Some High School, 4 = High School Graduate, 5 = Some College, 6 = College Graduate): "))
        Current_Smoker = int(input("Current Smoker (1 = Yes, 0 = No): "))
        Income_Category = int(input("Income Category (1 = <15K, 2 = 15K-25K, 3 = 25K-35K, 4 = 35K-50K, 5 = 50K+): "))
        Age_Group = int(input("Age Group (1 = 18-24, 2 = 25-34, 3 = 35-44, 4 = 45-54, 5 = 55-64, 6 = 65+): "))
        On_BP_Medication = int(input("On BP Medication (1 = Yes, 0 = No): "))
        
        # Return all inputs as a NumPy array
        user_input = np.array([HighChol, BMI, Diabetes, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, 
                               MentHlth, PhysHlth, DiffWalk, Sex, Education, Current_Smoker, Income_Category, 
                               Age_Group, On_BP_Medication]).reshape(1, -1)
        
        return user_input
    except ValueError as e:
        print("Invalid input. Please enter the correct data type.", e)
        return None

def load_model():
    """Loads the pre-trained ensemble model from the joblib file."""
    try:
        model = joblib.load('ensemble_model_with_tuning.joblib')
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print("The model file 'ensemble_model_with_tuning.joblib' was not found.")
        return None

def predict_heart_disease(model, user_input):
    """Makes a prediction using the loaded model and user input."""
    try:
        prediction = model.predict(user_input)
        
        if prediction[0] == 1:
            result = "Predicted Heart Disease Type: Coronary Heart Disease (CHD)"
        elif prediction[0] == 2:
            result = "Predicted Heart Disease Type: Myocardial Infarction (Heart Attack)"
        elif prediction[0] == 3:
            result = "Predicted Heart Disease Type: Stroke"
        else:
            result = "No Heart Disease Detected"
        
        return result
    except Exception as e:
        print("An error occurred during prediction.", e)
        return None

def main():
    """Main function to load the model, get user input, and make a prediction."""
    model = load_model()
    if model is not None:
        user_input = get_user_input()
        if user_input is not None:
            result = predict_heart_disease(model, user_input)
            print("\n", result)

if _name_ == "_main_":
    main()