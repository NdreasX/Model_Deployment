import pickle
import pandas as pd
import streamlit as st
from OOP import ModelTrainer

def main():
    st.title("Churn Prediction App")
    st.write("Upload a CSV file to predict churn")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read uploaded CSV file
        data_to_predict = pd.read_csv(uploaded_file)

        # Inisialisasi ModelTrainer
        trainer = ModelTrainer(numerical_features=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'],
                               categorical_features=['Geography', 'Gender'])

        # Lakukan preprocessing pada data yang akan diprediksi
        preprocessed_data, y = trainer.preprocess_data(data_to_predict)

        # Lakukan training model terbaik
        best_model = trainer.train_best_model(preprocessed_data, y)

        # Simpan model terbaik
        model_filename = 'best_model.pkl'
        trainer.save_model(model_filename)
        st.write("Model terbaik telah disimpan sebagai 'best_model.pkl'")

        # Lakukan prediksi churn
        predictions = best_model.predict(preprocessed_data)

        # Buat DataFrame untuk menampilkan hasil prediksi dalam bentuk tabel
        result_df = pd.DataFrame({'Sample': range(1, len(predictions)+1), 'Prediction': predictions})
        result_df['Prediction'] = result_df['Prediction'].map({0: 'No Churn', 1: 'Churn'})

        # Tampilkan hasil prediksi dalam tabel
        st.write("Predictions:")
        st.table(result_df)

if __name__ == "__main__":
    main()