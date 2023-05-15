# Import packages
import streamlit as st
import pandas as pd 
import plotly.express as px
import plotly.io as pio
#from PIL import Image
import matplotlib.pyplot as plt
import pickle 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st
import plotly.graph_objects as go

def calculate_bmi(weight, height):
    weight_kg = weight*0.454
    height_m = (height*2.54)/100
    bmi = weight_kg / (height_m ** 2)
    return bmi

def create_bmi_gauge(value):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "BMI"},
        gauge = {'axis': {'range': [None, 40]},
                 'bar': {'color': "darkblue"},
                 'steps' : [
                     {'range': [0, 18.5], 'color': 'lightgray'},
                     {'range': [18.5, 24.9], 'color': 'green'},
                     {'range': [24.9, 29.9], 'color': 'yellow'},
                     {'range': [29.9, 40], 'color': 'red'}],
                 'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': value}}))

    return fig

# Streamlit App
st.title("BMI Indicator Gauges")

weight = st.number_input("Enter your weight (in kg):", min_value=0.0)
height = st.number_input("Enter your height (in meters):", min_value=0.0)

if st.button("Submit"):
    bmi = calculate_bmi(weight, height)
    st.write("Your BMI:", bmi)

    # Create a grid of BMI indicator gauges
    cols = st.number_input("Enter the number of columns:", min_value=1, value=3)
    rows = (len(bmi) - 1) // cols + 1
    for i in range(rows):
        st.write("Row", i+1)
        for j in range(cols):
            index = i * cols + j
            if index < len(bmi):
                fig = create_bmi_gauge(bmi[index])
                st.plotly_chart(fig, use_container_width=True)
