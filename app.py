# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:14:38 2024

@author: Korisnik
"""

# load libraries
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to load and process the table
def load_excel(file):
    df = pd.read_excel(file)
    df.columns = ['Sample', 'Ratio', 'C']
    reference_ratio = (df.iloc[0]["Ratio"] + df.iloc[1]["Ratio"]) / 2
    # Compute the difference of each "Ratio" from the reference_ratio
    df['Diff'] = df['Ratio'] - reference_ratio
    return df

def get_mean_df(df):
    # find mean between the 2 points
    ratio_means = []
    samples = []
    concentrations = []

    for i in range(0, len(df) - 1, 2):
        sample_name = f"S{i // 2}"
    
        # Calculate the mean only for the "Ratio" column
        mean_ratio = df.iloc[i:i+2]["Ratio"].astype(float).mean() 
        concentration = df.iloc[i]["C"]  
    
        # Append values to lists
        samples.append(sample_name)
        ratio_means.append(mean_ratio)
        concentrations.append(concentration)

    # create new df with new values
    data = pd.DataFrame({
        'Sample': samples,
        'ratio': ratio_means,
        'C': concentrations
        })

    # compute ratio difference
    differences = []
    for i in range(0, len(data)):
        diff = data.iloc[i]["ratio"] - data.iloc[0]["ratio"] 
        differences.append(diff)
    
    data['Ratio'] = differences
    data = data.drop(['ratio'], axis = 1)
    return data


def weighted_LR(df, data):
    data['Weight'] = np.where(data['C'] != 0, 1 / (data['C']), 0)
    X  = data[['C']][1:]
    y = data['Ratio'][1:]
    model = LinearRegression()
    model.fit(X, y, sample_weight=data['Weight'][1:])
    #print(model.coef_[0], model.intercept_)
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)
    
    # Create a figures
    fig, ax = plt.subplots()
    ax.scatter(df[['C']][2:], df[['Diff']][2:], color='blue', label='Data')
    ax.plot(X, pred, color='red', label='Regression Line')
    ax.set_xlabel('C')
    ax.set_ylabel('Ratio')
    ax.text(0.40, 0.15, f"Ratio = {model.coef_[0]:.8f} * C + {model.intercept_:.8f}", transform=plt.gca().transAxes, fontsize=10)
    ax.text(0.70, 0.30, f"R² = {r2:.4f}", transform=plt.gca().transAxes, fontsize=10)
    #ax.text(0.70, 0.25, f"MSE = {mse:.4f}", transform=plt.gca().transAxes, fontsize=10)
    ax.legend()
    
    return fig, model


def get_final_df(df, model):
    
    c = (df['Diff'] - model.intercept_) / model.coef_[0]
    accuracy = (c / df['C']).replace([float('inf'), -float('inf')], None) * 100
    
    new_df = pd.DataFrame({
        'Sample': df['Sample'],
        'Real Ratio': df['Ratio'],
        'Real C': df['C'],
        'Shifted Ratio': df['Diff'],
        'Calculated C': c,
        'Accuracy (%)': accuracy})
    
    return new_df
    

# Initilize dict for storing coefficients
coeffs = {}
if "coeffs" not in st.session_state:
    st.session_state.coeffs = {}

st.title("Calibration")
option = st.radio("Choose an option:", ("Perform Calibration", "Get Concentration from Ratio"))

if option == "Perform Calibration":
    # Prompt for file upload only for calibration
    uploaded_file = st.file_uploader("Choose an XLSX file with 3 columns in order: sample name, ratio, concentration", type="xlsx")
    if uploaded_file is not None:
        # Process and display data
        df = load_excel(uploaded_file)
        data = get_mean_df(df)
    
        # Perform weighted linear regression and plot
        fig, model = weighted_LR(df, data)
        
        st.write("Processed Data:")
        new_df = get_final_df(df, model)
        st.dataframe(new_df)
        st.pyplot(fig)
        
        # Allow user to save new coefficients
        if st.button("Save Coefficients"):
            st.session_state.coeffs["beta"] = model.coef_[0]
            st.session_state.coeffs["alpha"] = model.intercept_
            st.session_state.coeffs["source_file"] = uploaded_file.name  # Save the name of the uploaded calibration file
            st.success(f"Coefficients calculated from file: {uploaded_file.name} are successfully saved.")
        
elif option == "Get Concentration from Ratio":
    if not st.session_state.coeffs:
        st.error("No coefficients found! Please perform calibration first.")
    else:
        uploaded_file = st.file_uploader("Choose an XLSX file with 2 columns in order: sample name, ratio", type="xlsx")
        
        if uploaded_file:
            coef = st.session_state.coeffs["beta"]
            intercept = st.session_state.coeffs["alpha"]
            
            df = pd.read_excel(uploaded_file)
            
            if df.shape[1] == 2:
                df.columns = ["Sample Name", "Ratio"]
                df["Concentration"] = (df["Ratio"] - intercept) / coef
                st.write(df)
            else:
                st.error("The uploaded file should contain exactly 2 columns: sample name and ratio!")


st.sidebar.write("### Current Calibration Details")
if st.session_state.coeffs:
    st.sidebar.write(f"Source File: {st.session_state.coeffs.get('source_file', 'Unknown')}")
    
    # Display the formula
    beta = st.session_state.coeffs.get("beta", 0)
    alpha = st.session_state.coeffs.get("alpha", 0)
    st.sidebar.write("Equation:")
    st.sidebar.latex(rf"Concentration = \frac{{Ratio - {alpha}}}{{{beta}}}")
else:
    st.sidebar.write("No coefficients saved yet.")