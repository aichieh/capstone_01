# Import packages
import streamlit as st
import pandas as pd 
import numpy as np
#import plotly.express as px
import plotly.io as pio
#from PIL import Image
import matplotlib.pyplot as plt
import pickle
import plotly.graph_objects as go
import pydeck as pdk

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Set the plotly template
pio.templates.default = "plotly_white"

# Setting the title of the tab and the favicon
st.set_page_config(page_title='Examining Health Care Data', page_icon = ':rain_cloud:', layout = 'centered')

# Setting the title on the page with some styling
st.markdown("<h1 style='text-align: center'>Examining Health Care Data</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)

# Read the data
df = pd.read_csv("./output/final_adult_try.csv")
if st.sidebar.checkbox("Display data", False):
    st.subheader("Show NHANES dataset")
    st.write(df)

st.sidebar.header('User Input Features')
st.sidebar.markdown("""
Input your data here .
""")
sex = df['gender']
age = df['age']
#weight = df['weight']
#height = df['height']
BMI = df['BMI']
#waist_circumference = df['waist_circumference']
systolic_bp = df['systolic_bp']
#heart_rate = df['heart_rate']
hypertension = df['hypertension']
take_HTN_medicine = df['take_HTN_medicine']
high_cholesterol = df['high_cholesterol']
take_HCL_medicine = df['take_HCL_medicine']
diabetes = df['diabetes']
heart_failure = df['heart_failure']
CAD = df['CAD']
angina = df['angina']
heart_attack = df['heart_attack']
#stroke = df['stroke']
sex_choice = st.sidebar.selectbox('Sex', ('Female', 'Male'))
age_choice = st.sidebar.slider('Age', 1, 100, 30)
BMI_choice = st.sidebar.slider('BMI (kg/m^2)', 15.0, 70.0, 23.0)
#weight_choice = st.sidebar.slider('Weight (lb)', 10.0, 400.0, 150.0)
#height_choice = st.sidebar.slider('Height (inch)', 10.0, 65.0, 80.0)
#waist_circumference_choice  = st.sidebar.slider('Waist Circumference (inch)', 10.0, 80.0, 30.0)
systolic_bp_choice = st.sidebar.slider('Blood Pressure(upper value) (mmHg)', 100.0, 250.0, 120.0)
#heart_rate_choice = st.sidebar.slider('Heart Rate (per minute)', 30.0, 150.0, 40.0)
hypertension_choice = st.sidebar.selectbox('Have hypertension', ('NO', 'YES'))
take_HTN_medicine_choice = st.sidebar.selectbox('Takes BP medicines', ('NO', 'YES'))
high_cholesterol_choice = st.sidebar.selectbox('Have high cholesterol', ('NO', 'YES'))
take_HCL_medicine_choice = st.sidebar.selectbox('Takes cholesterol medicines', ('NO', 'YES'))
diabetes_choice = st.sidebar.selectbox('Have diabetes', ('NO', 'YES'))
heart_failure_choice = st.sidebar.selectbox('Had any heart failure', ('NO', 'YES'))
CAD_choice = st.sidebar.selectbox('Had any coronary heart disease', ('NO', 'YES'))
angina_choice = st.sidebar.selectbox('Had any angina', ('NO', 'YES'))
heart_attack_choice = st.sidebar.selectbox('Had any heart attack', ('NO', 'YES'))
#stroke_choice = st.sidebar.selectbox('Had any prevalent Stroke', ('NO', 'YES'))

def value(lst, string):
    for i in range(len(lst)):
        if lst[i] == string:
            return i
sex=['Female', 'Male']
yn=['NO', 'YES']

#tab1, tab2 = st.tabs(["Stroke Risk", "Hypertension Risk"])

data = {'gender': value(sex, sex_choice),
                'age': age_choice,
#                'weight': weight_choice,
#                'height': height_choice,
                'BMI': BMI_choice,
#                'waist_circumference': waist_circumference_choice,
                'systolic_bp': systolic_bp_choice,
                'hypertension': value(yn, hypertension_choice),
                'take_HTN_medicine': value(yn, take_HTN_medicine_choice),
                'high_cholesterol': value(yn, high_cholesterol_choice),
                'take_HCL_medicine': value(yn, take_HCL_medicine_choice),
#                'heart_rate': heart_rate_choice,              
                'diabetes': value(yn, diabetes_choice),
                'heart_failure': value(yn, heart_failure_choice),
                'CAD': value(yn, CAD_choice),
                'angina': value(yn, angina_choice),
                'heart_attack': value(yn, heart_attack_choice),
#                'stroke': value(yn, stroke_choice)
               }
features = np.array(pd.DataFrame(data, index=[0]))
#st.write(features)

#data_load_state1.text("Predicting...")
# Reads in saved classification model
model = pickle.load(open('stroke_adult.pkl', 'rb'))
           
# Apply model to make predictions
prediction = model.predict(features)
prediction_proba = model.predict_proba(features).reshape(2,)
#st.write("Risk of Hypertension") 
risk = (prediction_proba[1]*100).round(2)
#st.write(risk, " %")

@st.cache(allow_output_mutation=True)
def userData():
    return []

@st.cache(allow_output_mutation=True)
def delta(l, p):
    if len(l) == 0:
        l.extend([0, (p)])
        d = 0
    else:
        l.pop(0)
        l.append((p).round(2))
        d = l[1] - l[0]
    return d

#col1, col2 = st.columns(2)
st.metric(
    label="Risk of Stroke", 
    value= str(risk) + " %", 
    delta=str(delta(userData(), risk)) + " percentage points", 
    help="""
    The change in percentage points is displayed below.
    """,
    delta_color ="inverse"
)
#######BMI Indicator##################

@st.cache(allow_output_mutation=True)
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

#######Hypertension Indicator##################    
    
# Hypertension Staging Indicator Gauge
def assesHTN(systolic, diastolic):
    if systolic == 0 or diastolic == 0:
        inf = """
        Note: Information is unreliable.
        Enter both systolic and diastolic blood pressure to see the hypertension staging gauge.
        """
    elif systolic < 120 and diastolic < 80:
        inf = "Stage of Hypertension:\nNormal"
    elif systolic < 130 and diastolic < 80:
        inf = "Stage of Hypertension:\nPrehypertension"
    elif systolic < 140 or diastolic < 90:
        inf = "Stage of Hypertension:\nStage I Hypertension"
    elif systolic >= 140 or diastolic >= 90:
        inf = "Stage of Hypertension:\nStage II Hypertension"
    elif systolic > 180 or diastolic > 120:
        inf = "Stage of Hypertension:\nHypertensive Crisis"
    return inf

def plot_systolic_gauge(value):
    plot_gauge("Systolic Blood Pressure", value, 0, 200)

def plot_diastolic_gauge(value):
    plot_gauge("Diastolic Blood Pressure", value, 0, 120)

def plot_gauge(title, value, min_value, max_value):
    gauge_layer = pdk.Layer(
        "GaugeLayer",
        data=[{"value": value}],
        get_value="value",
        gauge_color_namespace="value",
        gauge_color_scale=[
            [min_value, "green"],
            [(min_value + max_value) / 2, "yellow"],
            [max_value, "red"],
        ],
        gauge_radius=0.95,
        gauge_inner_radius=0.75,
        pickable=False,
        auto_highlight=False,
    )

    view_state = pdk.ViewState(
        latitude=0,
        longitude=0,
        zoom=0,
        bearing=0,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=[gauge_layer],
        initial_view_state=view_state,
        views=[pdk.View(type="GaugeView")],
    )

    st.pydeck_chart(deck)

systolic = st.number_input("Enter your Systolic Blood Pressure(upper value) (mmHg):", min_value=0.0)
diastolic = st.number_input("Enter your Diastolic Blood Pressure(lower value) (mmHg)", min_value=0.0)

# Streamlit App
if st.button("check it out"):
    st.title("Blood Pressure Gauges")
    st.subheader("Systolic Blood Pressure")
    #plot_systolic_gauge(systolic_bp)
    create_bmi_gauge(systolic_bp)
    st.subheader("Diastolic Blood Pressure")
    #plot_diastolic_gauge(diastolic_bp)
    create_bmi_gauge(diastolic_bp)
