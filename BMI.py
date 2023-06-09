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

@st.cache(allow_output_mutation=True)
def assesBMI(BMI):
    if BMI > 45:
        inf = """
        Note: Information is unreliable.
        BMI > 45.
        """
    elif BMI <= 10:
        inf = "BMI level:\nBMI too low"
    elif BMI < 18.5:
        inf = "BMI level:\nUnderweight"
    elif BMI >= 18.5 and BMI < 25:
        inf = "BMI level:\nNormal Weight"
    elif BMI >= 25 and BMI < 30:
        inf = "BMI level:\nOverweight"
    elif BMI >= 30 and BMI < 35:
        inf = "BMI level:\nModerate Obesity"
    elif BMI >= 35 and BMI < 40:
        inf = "BMI level:\nStrong Obesity"
    elif BMI >= 40:
        inf = "BMI level:\nExtreme Obesity"
    return inf

def create_bmi_gauge(value):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
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
                 'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 30}}))

    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
    return fig

# Streamlit App
st.title("BMI Indicator Gauges")

weight = st.number_input("Enter your weight (in lb):", min_value=0.0)
height = st.number_input("Enter your height (in inch):", min_value=0.0)

if st.button("Calculate BMI"):
    bmi = calculate_bmi(weight, height)
    st.write("Your BMI:", bmi)
    st.write(assesBMI(bmi))
    # Display BMI gauge for the submitted entry
    st.plotly_chart(create_bmi_gauge(bmi))

#fig = create_bmi_gauge(bmi[index])
#st.plotly_chart(fig, use_container_width=True)
