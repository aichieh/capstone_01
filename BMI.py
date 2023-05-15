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

# Create subplots with grid layout
fig = make_subplots(rows=2, cols=2, subplot_titles=("Person 1", "Person 2", "Person 3", "Person 4"))

# Create subplots with grid layout
fig = make_subplots(rows=2, cols=2, subplot_titles=("Person 1", "Person 2", "Person 3", "Person 4"))

# Define BMI data for each person
bmi_data = [20.5, 22.1, 25.8, 23.9]

# Iterate over each subplot and add BMI indicator gauge
for i, bmi in enumerate(bmi_data):
    row = i // 2 + 1
    col = i % 2 + 1

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=bmi,
            title="BMI",
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'visible': False}},
        ),
        row=row, col=col
    )

# Update layout and display the plot
fig.update_layout(height=600, width=800, title="BMI Indicators")
fig.show()
