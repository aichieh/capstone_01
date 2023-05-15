# Import packages
import streamlit as st
import pandas as pd 
import numpy as np
#import plotly.express as px
import plotly.io as pio
#from PIL import Image
import matplotlib.pyplot as plt
import pickle 

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
weight_choice = st.sidebar.slider('Weight (lb)', 10.0, 400.0, 150.0)
height_choice = st.sidebar.slider('Height (inch)', 10.0, 65.0, 80.0)
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
#######Additional Information##################

@st.cache(allow_output_mutation=True)
def assesBMI(BMI, AGE):
    if BMI > 45 and AGE > 75:
        inf = """
        Note: Information is unreliable.
        BMI > 45 and age > 75.
        """
    elif BMI <= 10:
        inf = "BMI level:\nBMI too low"
    elif BMI < 18.5 and BMI > 10:
        inf = "BMI level:\nShortweight"
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

st.text(assesBMI(BMI_choice, age_choice))

# BMI Meter
#@st.cache(allow_output_mutation=True)
#st.title("BMI Meter")
#def calculate_bmi(weight, height):
#    weight_kg = weight*0.454
#    height_m = (height*2.54)/100
#    bmi = weight_kg / (height_m ** 2)
#    return bmi

#check = st.sidebar.button('Submit')
#if(check):
#    bmi = calculate_bmi(weight, height) 
#    st.title(f'Your BMI : {bmi}')
#    if bmi <18.5:
#        st.title("You are Underweight")
#    elif bmi>= 18.5 and bmi<25:
#        st.title("You are Normal")
#    elif bmi >=25 and bmi<30:
#        st.title("You are Overweight")
#    else:
#        st.title("You are Obese")
#        
#def create_bmi_gauge(bmi_value):
#    fig = go.Figure(go.Indicator(
#        mode = "gauge+number",
#        value = bmi_value,
#        domain = {'x': [0, 1], 'y': [0, 1]},
#        title = {'text': "BMI"},
#        gauge = {'axis': {'range': [None, 40]},
#                 'bar': {'color': "darkblue"},
#                 'steps' : [
#                     {'range': [0, 18.5], 'color': 'lightgray'},
#                     {'range': [18.5, 24.9], 'color': 'green'},
#                     {'range': [24.9, 29.9], 'color': 'yellow'},
#                     {'range': [29.9, 40], 'color': 'red'}],
#                 'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': value}}))
#    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
#    return fig
# Reads in saved classification model
#model2 = pickle.load(open('htn.pkl', 'rb'))


#data2 = {'weight': weight_choice,
#        'height': height_choice,
#        'BMI': BMI_choice,
#        'waist_circumference': waist_circumference_choice,
#        #'hypertension': value(yn, hypertension_choice),
#        'take_HTN_medicine': value(yn, take_HTN_medicine_choice),
#        'high_cholesterol': value(yn, high_cholesterol_choice),
#        'take_HCL_medicine': value(yn, take_HCL_medicine_choice),
#        'heart_rate': heart_rate_choice,
#        'systolic_bp': systolic_bp_choice,
#        'gender': value(sex, sex_choice),
#        'age': age_choice,
#        'diabetes': value(yn, diabetes_choice),
#        'heart_failure': value(yn, heart_failure_choice),
#        'CAD': value(yn, CAD_choice),
#        'angina': value(yn, angina_choice),
#        'heart_attack': value(yn, heart_attack_choice),
#        'stroke': value(yn, stroke_choice)
#       }
#features2 = np.array(pd.DataFrame(data2, index=[0]))

# Apply model to make predictions
#prediction2 = model2.predict(features2)
#prediction_proba2 = model2.predict_proba(features2).reshape(2,)
#st.write("Risk of Hypertension") 
#risk2 = (prediction_proba[1]*100).round(2)
#st.write(risk2, " %")

#col2.metric(
#    label="Risk of Hypertension", 
#    value= str(risk2) + " %", 
#    delta=str(delta(userData(), risk2)) + " percentage points", 
#    help="""
#    The change in percentage points is displayed below.
#    """,
#    delta_color ="inverse"
#)

