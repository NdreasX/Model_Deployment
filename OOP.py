import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class ModelTrainer:
    def __init__(self, numerical_features, categorical_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.best_model = None
    
    def preprocess_data(self, data):
        # Memisahkan fitur numerik dan kategorikal
        X = data[self.numerical_features + self.categorical_features]
        y = data['churn']
        
        # Proses imputasi untuk nilai yang hilang pada fitur numerik
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Mengisi nilai yang hilang dengan rata-rata
            ('scaler', StandardScaler())  # Melakukan penskalaan fitur
        ])

        # Proses one-hot encoding untuk fitur kategorikal
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Mengisi nilai yang hilang dengan 'missing'
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Melakukan one-hot encoding
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        # Lakukan preprocessing pada data yang akan diprediksi
        preprocessed_data = preprocessor.fit_transform(X)
        
        return preprocessed_data, y
    
    def train_best_model(self, X_train, y_train):
        # Split data menjadi data latih dan data uji
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_f1 = f1_score(y_test, rf_pred)

        # Train XGBoost model
        xgb_model = XGBClassifier(random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_f1 = f1_score(y_test, xgb_pred)

        # Select the best model
        if rf_f1 > xgb_f1:
            self.best_model = rf_model
        else:
            self.best_model = xgb_model
        
        return self.best_model

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.best_model, file)