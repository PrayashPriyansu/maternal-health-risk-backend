import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class MaternalHealthRiskModel:
    def __init__(self, data_path):
        import os
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The file {data_path} does not exist.")
        
        self.data = pd.read_csv(data_path)
        if self.data.empty:
            raise ValueError("The CSV file is empty.")

        self.X_train = None
        self.y_train = None
        self.models = {}
        self.best_model = None
        self.label_encoder = None

    @staticmethod
    def categorize_blood_pressure(systolic, diastolic):
        """Categorize blood pressure levels."""
        if systolic < 120 and diastolic < 80:
            return 0
        elif 120 <= systolic < 130 and 80 <= diastolic < 85:
            return 1
        elif (130 <= systolic < 140 or 85 <= diastolic < 90):
            return 2
        else:
            return 3

    @staticmethod
    def categorize_age(age):
        """Categorize age into groups."""
        if age < 20:
            return 0
        elif 20 <= age < 35:
            return 1
        elif 35 <= age < 45:
            return 2
        else:
            return 3

    def preprocess_data(self):
        """Preprocess the data for training."""
        self.data['BP_Category'] = self.data.apply(
            lambda row: self.categorize_blood_pressure(row['SystolicBP'], row['DiastolicBP']), axis=1)
        self.data['Age_Group'] = self.data['Age'].apply(self.categorize_age)
        
        X = self.data.drop(columns=['RiskLevel'])
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(self.data['RiskLevel'])
        
        self.X_train, _, self.y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return self

    def create_models(self):
        """Create pipelines for different machine learning models."""
        self.models = {
            'Logistic Regression': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=1000))
            ]),
            'Random Forest': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('classifier', RandomForestClassifier(random_state=42))
            ]),
            'Gradient Boosting': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('classifier', GradientBoostingClassifier(random_state=42))
            ]),
            'SVM': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', SVC(probability=True, random_state=42))
            ])
        }
        return self

    def train_models(self):
        """Train all models and identify the best-performing one."""
        model_performance = {}
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_train)  # Train accuracy
            accuracy = accuracy_score(self.y_train, y_pred)
            model_performance[name] = accuracy
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        self.best_model = max(model_performance, key=model_performance.get)
        print(f"Best Model: {self.best_model}")
        return self

    def predict(self, input_data):
        """Predict risk levels for new patient data."""
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        input_data['BP_Category'] = input_data.apply(
            lambda row: self.categorize_blood_pressure(row['SystolicBP'], row['DiastolicBP']), axis=1)
        input_data['Age_Group'] = input_data['Age'].apply(self.categorize_age)
        
        expected_columns = self.X_train.columns
        for col in expected_columns:
            if col not in input_data:
                input_data[col] = np.nan  # Fill missing columns with NaN

        input_data = input_data[expected_columns]

        best_model = self.models[self.best_model]
        predictions = best_model.predict(input_data)
        predicted_probabilities = best_model.predict_proba(input_data).max(axis=1)
        
        results = {
            "Predicted Risk Level": self.label_encoder.inverse_transform(predictions)[0],
            "Prediction Probability": predicted_probabilities[0]
        }
        return results

    def run_analysis(self):
        """Run the entire analysis pipeline."""
        self.preprocess_data().create_models().train_models()
        return self
