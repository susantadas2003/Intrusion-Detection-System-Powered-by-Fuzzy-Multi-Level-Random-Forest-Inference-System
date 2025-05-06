import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from collections import Counter
import io

# Page config
st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# Load saved models and encoders
model = pickle.load(open('intrusion_model.sav', 'rb'))
selected_indices = pickle.load(open('selected_indices.sav', 'rb'))
label_encoder = pickle.load(open('label_encoder.sav', 'rb'))

scaler = StandardScaler()

def preprocess_input(df_input):
    df = df_input.copy()

    # Encode categorical columns with factorize to avoid label mismatch
    categorical_columns = ['protocol_type', 'service', 'flag']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = pd.factorize(df[col])[0]

    # Scale and select features
    df_scaled = scaler.fit_transform(df)
    df_selected = df_scaled[:, selected_indices]
    return df_selected

def assess_severity(prob):
    if prob < 0.5:
        return "Low"
    elif 0.5 <= prob < 0.8:
        return "Medium"
    else:
        return "High"

st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .reportview-container .main .block-container{
            padding-top: 2rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 2rem;
        }
        h1 {
            color: #0a5275;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔐 Intrusion Detection System")
st.markdown("Upload your network traffic CSV file (without labels) to detect intrusions using a Multi-Level Random Forest model enhanced by a Fuzzy Inference System.")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type="csv")

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(df_input.head(), use_container_width=True)

    X_final = preprocess_input(df_input)

    # Predict labels
    predictions = model.predict(X_final)
    predicted_labels = label_encoder.inverse_transform(predictions)
    probs = model.predict_proba(X_final)[:, 1]
    severity_levels = [assess_severity(p) for p in probs]

    df_output = df_input.copy()
    df_output['Predicted_Label'] = predicted_labels
    df_output['Severity'] = severity_levels

    st.subheader("📊 Prediction Results")
    st.dataframe(df_output.head(), use_container_width=True)

    # Plot attack distribution
    st.subheader("📈 Attack Type Distribution")
    label_counts = Counter(predicted_labels)
    labels = list(label_counts.keys())
    sizes = list(label_counts.values())
    fig1, ax1 = plt.subplots()
    colors = sns.color_palette('pastel')[0:len(labels)]
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Download output
    st.subheader("⬇️ Download Predictions")
    csv_buffer = io.StringIO()
    df_output.to_csv(csv_buffer, index=False)
    st.download_button("Download CSV", csv_buffer.getvalue(), file_name="predictions_output.csv", mime='text/csv')
