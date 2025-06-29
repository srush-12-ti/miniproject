# miniproject
This project uses machine learning to predict heart disease risk based on lifestyle and health factors like age, BMI, smoking, alcohol use, diet, exercise, and conditions such as diabetes or cancer. After preprocessing and balancing the data using SMOTE, several models were trained and combined into an ensemble meta-model. 

Project structure:
heart-disease-prediction/
│
├── data/
│   ├── raw/                     
│   └── processed/               
│
├── models/
│   └── ensemble_model.joblib    
│
├── notebooks/
│   └── exploration.ipynb       
│   └── model_training.ipynb     
│
├── outputs/
│   ├── evaluation/              
│   └── reports/                
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py            
│   ├── model_utils.py           
│   └── visualization.py         
│
├── requirements.txt            
├── .gitignore                   
├── README.md                    
└── main.py                      


# Heart Disease Prediction

This project predicts heart disease risk using machine learning models based on lifestyle and medical data.

## Features
- Lifestyle factors (smoking, drinking, physical activity)
- Medical conditions (diabetes, cancer)
- Data preprocessing, binning, SMOTE
- Ensemble modeling
- Evaluation and visualizations

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python main.py`
