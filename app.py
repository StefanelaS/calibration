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


def weighted_LR(df):
    df['Weight'] = np.where(df['C'] != 0, 1 / (df['C']), 0)
    X  = df[['C']][1:]
    y = df['Ratio'][1:]
    model = LinearRegression()
    model.fit(X, y, sample_weight=df['Weight'][1:])
    #print(model.coef_[0], model.intercept_)
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X, pred , color='red', label='Regression Line')
    
    # Create a figures
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Data')
    ax.plot(X, pred, color='red', label='Regression Line')
    ax.set_xlabel('C')
    ax.set_ylabel('Ratio')
    ax.text(0.40, 0.15, f"Ratio = {model.coef_[0]:.8f} * C + {model.intercept_:.8f}", transform=plt.gca().transAxes, fontsize=10)
    ax.text(0.70, 0.30, f"R² = {r2:.4f}", transform=plt.gca().transAxes, fontsize=10)
    ax.text(0.70, 0.25, f"MSE = {mse:.4f}", transform=plt.gca().transAxes, fontsize=10)
    ax.legend()
    
    return fig

df = load_excel("Calibration.xlsx")
data = get_mean_df(df)
fig = weighted_LR(data)

names = {
    "Tryptophan": {"beta": 0.001115585071256631, "alpha": 1.0257405790000629}
}

st.title("Calibration")
option = st.radio("Choose an option:", ("Perform Calibration", "Get Concentration from Ratio"))

if option == "Perform Calibration":
    # Prompt for file upload only for calibration
    uploaded_file = st.file_uploader("Choose an XLSX file", type="xlsx")
    if uploaded_file is not None:
        # Process and display data
        df = load_excel(uploaded_file)
        data = get_mean_df(df)
        st.write("Processed Data:")
        st.dataframe(data)
        
        # Perform weighted linear regression and plot
        fig = weighted_LR(data)
        st.pyplot(fig)
        
elif option == "Get Concentration from Ratio":
    # For concentration estimation, prompt for ratio input directly
    compound_name = st.selectbox("Select the compound:", list(names.keys()))
    input_ratio = st.number_input("Enter a ratio value:")
    
    if input_ratio and compound_name:
        # Retrieve the model parameters for the selected compound
        coef = names[compound_name]["beta"]
        intercept = names[compound_name]["alpha"]
        
        # Calculate the concentration
        C = (input_ratio - intercept) / coef
        st.write(f"Estimated concentration: {C:.6f}")