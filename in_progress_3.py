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

# Set the plotly template
pio.templates.default = "plotly_white"

# Setting the title of the tab and the favicon
st.set_page_config(page_title='Examining Hypertension Using Health Care Data', page_icon = ':rain_cloud:', layout = 'centered')

# Setting the title on the page with some styling
st.markdown("<h1 style='text-align: center'>Examining Hypertension Using Health Care Data</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)

# Read the data
df = pd.read_csv("./output/df_stroke.csv")
if st.sidebar.checkbox("Display data", False):
    st.subheader("Show NHANES dataset")
    st.write(df)

st.sidebar.header('User Input Features')
st.sidebar.markdown("""
Input your data here .
""")
sex = df['gender']
age = df['age']
weight = df['weight']
height = df['height']
#waist_circumference = df['waist_circumference']
systolic_bp = df['systolic_bp']
#heart_rate = df['heart_rate']
#BMI = df['BMI']
#hypertension = df['hypertension']
take_HTN_medicine = df['take_HTN_medicine']
high_cholesterol = df['high_cholesterol']
take_HCL_medicine = df['take_HCL_medicine']
diabetes = df['diabetes']
#stroke = df['stroke']
#heart_failure = df['heart_failure']
#CAD = df['CAD']
#angina = df['angina']
#heart_attack = df['heart_attack']
sex_choice = st.sidebar.selectbox('Sex', ('Female', 'Male'))
age_choice = st.sidebar.slider('Age', 1, 100, 30)
weight_choice = st.sidebar.slider('Weight (lb)', 10.0, 400.0, 150.0)
height_choice = st.sidebar.slider('Height (inch)', 10.0, 65.0, 80.0)
#waist_circumference_choice  = st.sidebar.slider('Waist Circumference (inch)', 10.0, 80.0, 30.0)
systolic_bp_choice = st.sidebar.slider('Blood Pressure(upper value) (mmHg)', 100.0, 250.0, 120.0)
#heart_rate_choice = st.sidebar.slider('Heart Rate (per minute)', 30.0, 150.0, 40.0)
#BMI_choice = st.sidebar.slider('BMI (kg/m^2)', 15.0, 70.0, 23.0)
#hypertension_choice = st.sidebar.selectbox('Have hypertension', ('NO', 'YES'))
take_HTN_medicine_choice = st.sidebar.selectbox('Takes BP medicines', ('NO', 'YES'))
high_cholesterol_choice = st.sidebar.selectbox('Have high cholesterol', ('NO', 'YES'))
take_HCL_medicine_choice = st.sidebar.selectbox('Takes cholesterol medicines', ('NO', 'YES'))
diabetes_choice = st.sidebar.selectbox('Have diabetes', ('NO', 'YES'))
#stroke_choice = st.sidebar.selectbox('Had any prevalent Stroke', ('NO', 'YES'))
#heart_failure_choice = st.sidebar.selectbox('Had any heart failure', ('NO', 'YES'))
#CAD_choice = st.sidebar.selectbox('Had any coronary heart disease', ('NO', 'YES'))
#angina_choice = st.sidebar.selectbox('Had any angina', ('NO', 'YES'))
#heart_attack_choice = st.sidebar.selectbox('Had any heart attack', ('NO', 'YES'))

#engine_choice = st.sidebar.selectbox('', engines)
# Creating the container for the first plot
#with st.beta_expander('Stroke Prediction'):

# Creating a selectbox dropdown with the categorical features to choose from
#    cat_option = st.selectbox('Select a feature to examine', cat_cols, key='cat_cols1')

def value(lst, string):
    for i in range(len(lst)):
        if lst[i] == string:
            return i
sex=['Female', 'Male']
yn=['NO', 'YES']

tab1, tab2 = st.tabs(["Basic Health Information", "Chronic Disease Risk"])
  

if st.sidebar.button('Submit'):
        data = {'weight': weight_choice,
                'height': height_choice,
#                'BMI': BMI_choice,
#                'waist_circumference': waist_circumference_choice,
#                'hypertension': value(yn, hypertension_choice),
                'take_HTN_medicine': value(yn, take_HTN_medicine_choice),
                'high_cholesterol': value(yn, high_cholesterol_choice),
                'take_HCL_medicine': value(yn, take_HCL_medicine_choice),
#                'heart_rate': heart_rate_choice,
                'systolic_bp': systolic_bp_choice,
                'gender': value(sex, sex_choice),
                'age': age_choice,
                'diabetes': value(yn, diabetes_choice),
#                'heart_failure': value(yn, heart_failure_choice),
#                'CAD': value(yn, CAD_choice),
#                'angina': value(yn, angina_choice),
#                'heart_attack': value(yn, heart_attack_choice),
#                'stroke': value(yn, stroke_choice)
               }
        features = pd.DataFrame(data, index=[0])
        #features = ['weight','height','BMI','waist_circumference','hypertension','take_HTN_medicine','high_cholesterol','take_HCL_medicine','heart_rate','systolic_bp','gender','age',
       #             'diabetes','heart_failure','CAD','angina','heart_attack','stroke']
        #df = df[features]
        st.write(features)

        st.markdown("<h2 style='text-align: center; color:#000066;'>Data gathered........</h2>", unsafe_allow_html = True)
        st.markdown("<h2 style='text-align: center; color:#000066;'>Processing Results........</h2>", unsafe_allow_html = True)
        # Reads in saved classification model
        model = pickle.load(open('stroke.pkl', 'rb'))
           
        # Apply model to make predictions
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features).reshape(2,)
        st.write("Risk of Stroke")
        risk = (prediction_proba[1]*100).round(2) 
        st.write(risk, " %")

        yes = prediction_proba[1]
        no = prediction_proba[0]     
        
        if prediction==0:
            st.error('Warning! You have high risk of getting a heart attack!')
        else:
            st.success('You have lower risk of getting a heart disease!')
    
###-------------------------------------------------------------------------------------------------------------------------------------------------------------------####
# BMI Meter
def calculate_bmi(weight, height):
    weight_kg = weight*2.205
    height_m = (height*2.54)/100
    bmi = weight_kg / (height_m ** 2)
    return bmi

st.title("BMI Meter")
weight = st.number_input("Enter your weight:", min_value=0.0)
height = st.number_input("Enter your height:", min_value=0.0)
if weight > 0 and height > 0:
    bmi = calculate_bmi(weight, height)
    st.write("Your BMI:", bmi)

# Hypertension Stage Meter
def get_hypertension_stage(systolic, diastolic):
    if systolic >= 180 or diastolic >= 120:
        return "Hypertensive Crisis"
    elif systolic >= 140 or diastolic >= 90:
        return "Stage 2 Hypertension"
    elif systolic >= 130 or diastolic >= 80:
        return "Stage 1 Hypertension"
    elif systolic >= 120 and diastolic < 80:
        return "Elevated"
    else:
        return "Normal"

st.title("Hypertension Stage Meter")
systolic = st.number_input("Enter your systolic blood pressure:", min_value=0)
diastolic = st.number_input("Enter your diastolic blood pressure:", min_value=0)
if systolic > 0 and diastolic > 0:
    stage = get_hypertension_stage(systolic, diastolic)
    st.write("Your Hypertension Stage:", stage)

# Diabetes Stage Meter
def get_diabetes_stage(fasting_glucose):
    if fasting_glucose >= 126:
        return "Diabetes"
    elif fasting_glucose >= 100:
        return "Prediabetes"
    else:
        return "Normal"
       

def preprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal ):   
 
    
    # Pre-processing user input   
    if sex=="male":
        sex=1 
    else: sex=0
    
    
    if cp=="Typical angina":
        cp=0
    elif cp=="Atypical angina":
        cp=1
    elif cp=="Non-anginal pain":
        cp=2
    elif cp=="Asymptomatic":
        cp=2
    
    if exang=="Yes":
        exang=1
    elif exang=="No":
        exang=0
 
    if fbs=="Yes":
        fbs=1
    elif fbs=="No":
        fbs=0
 
    if slope=="Upsloping: better heart rate with excercise(uncommon)":
        slope=0
    elif slope=="Flatsloping: minimal change(typical healthy heart)":
          slope=1
    elif slope=="Downsloping: signs of unhealthy heart":
        slope=2  
 
    if thal=="fixed defect: used to be defect but ok now":
        thal=6
    elif thal=="reversable defect: no proper blood movement when excercising":
        thal=7
    elif thal=="normal":
        thal=2.31

    if restecg=="Nothing to note":
        restecg=0
    elif restecg=="ST-T Wave abnormality":
        restecg=1
    elif restecg=="Possible or definite left ventricular hypertrophy":
        restecg=2


    user_input=[age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal]
    user_input=np.array(user_input)
    user_input=user_input.reshape(1,-1)
    user_input=scal.fit_transform(user_input)
    prediction = model.predict(user_input)

    return prediction

    

       
    # front end elements of the web page 
html_temp = """ 
    <div style ="background-color:pink;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Healthy Heart App</h1> 
    </div> 
    """
      
# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True) 
st.subheader('by Amlan Mohanty ')
      
# following lines create boxes in which user can enter data required to make prediction
age=st.selectbox ("Age",range(1,121,1))
sex = st.radio("Select Gender: ", ('male', 'female'))
cp = st.selectbox('Chest Pain Type',("Typical angina","Atypical angina","Non-anginal pain","Asymptomatic")) 
trestbps=st.selectbox('Resting Blood Sugar',range(1,500,1))
restecg=st.selectbox('Resting Electrocardiographic Results',("Nothing to note","ST-T Wave abnormality","Possible or definite left ventricular hypertrophy"))
chol=st.selectbox('Serum Cholestoral in mg/dl',range(1,1000,1))
fbs=st.radio("Fasting Blood Sugar higher than 120 mg/dl", ['Yes','No'])
thalach=st.selectbox('Maximum Heart Rate Achieved',range(1,300,1))
exang=st.selectbox('Exercise Induced Angina',["Yes","No"])
oldpeak=st.number_input('Oldpeak')
slope = st.selectbox('Heart Rate Slope',("Upsloping: better heart rate with excercise(uncommon)","Flatsloping: minimal change(typical healthy heart)","Downsloping: signs of unhealthy heart"))
ca=st.selectbox('Number of Major Vessels Colored by Flourosopy',range(0,5,1))
thal=st.selectbox('Thalium Stress Result',range(1,8,1))



#user_input=preprocess(sex,cp,exang, fbs, slope, thal )
pred=preprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal)




if st.button("Predict"):    
  if pred[0] == 0:
    st.error('Warning! You have high risk of getting a heart attack!')
    
  else:
    st.success('You have lower risk of getting a heart disease!')
    
   



st.sidebar.subheader("About App")

st.sidebar.info("This web app is helps you to find out whether you are at a risk of developing a heart disease.")
st.sidebar.info("Enter the required fields and click on the 'Predict' button to check whether you have a healthy heart")
st.sidebar.info("Don't forget to rate this app")
