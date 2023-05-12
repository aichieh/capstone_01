# Import packages
import streamlit as st
import pandas as pd 
import plotly.express as px
import plotly.io as pio
#from PIL import Image
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

def value(lst, string):
    for i in range(len(lst)):
        if lst[i] == string:
            return i
sex=['Female', 'Male']
#edu=['10th pass', '12th pass/Diploma', 'Bachelors', 'Masters or Higher']
yn=['NO', 'YES']

# Set the plotly template
pio.templates.default = "plotly_white"

# Setting the title of the tab and the favicon
st.set_page_config(page_title='Examining Hypertension Using Health Care Data', page_icon = ':rain_cloud:', layout = 'centered')

# Setting the title on the page with some styling
st.markdown("<h1 style='text-align: center'>Examining Hypertension Using Health Care Data</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)

# Read the data
#df1 = pd.read_csv("./output/df_try.csv")
# Creating the container for the first plot
#with st.beta_expander('Stroke Prediction'):

# Creating a selectbox dropdown with the categorical features to choose from
#    cat_option = st.selectbox('Select a feature to examine', cat_cols, key='cat_cols1')

# The code to run the first plot

#Predict training set:
#y_pred = lg.predict(X_test)

#CM = confusion_matrix(y_test, y_pred)
#TN = CM[0][0]
#FN = CM[1][0]
#TP = CM[1][1]
#FP = CM[0][1]

#result=pd.DataFrame()
# Sensitivity, hit rate, recall, or true positive rate
#result['TPR'] = [round(TP/(TP+FN),2)]
 # Specificity or true negative rate
#result['TNR'] = [round(TN/(TN+FP),2) ]
        # Fall out or false positive rate
#result['FPR'] = [round(FP/(FP+TN),2)]
         # False negative rate
#result['FNR'] = [round(FN/(TP+FN),2)]

#lg_probs = lg.predict_proba(X_test)
        # keep probabilities for the positive outcome only
#lg_probs = lg_probs[:, 1]
#result['AUC'] = [round(roc_auc_score(y_test, lg_probs),2)]

# Explaination of the features displays along with the graph
#st.markdown('**Explaination of the feature selected:**')

st.sidebar.header('User Input Features')
st.sidebar.markdown("""
Input your data here.
""")
gender = st.sidebar.slider('Sex', ('NO', 'YES'))
age = st.sidebar.slider('Age', 1, 100, 30)
weight = st.sidebar.selectbox('Weight (lb)', 10.0, 400.0, 150.0)
waist_circumference = st.sidebar.selectbox('Waist Circumference (inch)', 10.0, 80.0, 30.0)
systolic_bp = st.sidebar.slider('Blood Pressure(upper value) (mmHg)', 100.0, 250.0, 120.0)
BMI = st.sidebar.slider('BMI (kg/m^2)', 15.0, 70.0, 23.0)

hypertension = st.sidebar.selectbox('Have hypertension', ('NO', 'YES'))
take_HTN_medicine = st.sidebar.selectbox('Takes BP medicines', ('NO', 'YES'))
high_cholesterol = st.sidebar.slider('Have high cholesterol', ('NO', 'YES'))
take_HCL_medicine = st.sidebar.selectbox('Takes cholesterol medicines', ('NO', 'YES'))
diabetes = st.sidebar.selectbox('Have diabetes', ('NO', 'YES'))
stroke = st.sidebar.selectbox('Had any prevalent Stroke', ('NO', 'YES'))
heart_failure = st.sidebar.selectbox('Had any heart failure', ('NO', 'YES'))
coronary_heart_disease = st.sidebar.selectbox('Had any heart failure', ('NO', 'YES'))
angina = st.sidebar.selectbox('Had any angina', ('NO', 'YES'))
heart_attack = st.sidebar.selectbox('Had any heart attack', ('NO', 'YES'))

#60_sec_pulse = st.sidebar.slider('Heart Rate (per minute)', 30.0, 130.0, 40.0)
#glucose = st.sidebar.slider('Glucose (mg/dl)', 100.0, 500.0, 110.0)
#education = st.sidebar.selectbox('Education', ('10th pass', '12th pass/Diploma', 'Bachelors', 'Masters or Higher'))
#current_smoker = st.sidebar.selectbox('Current Smoker', ('NO', 'YES'))
#cigsPerDay = st.sidebar.slider('Cigarettes per Day', 0, 100, 20)


st.markdown("<h3 style='text-align: center; color:#4dffa6;'>Update your details in the sidebar</h3>", unsafe_allow_html = True)
st.markdown("<h3 style='text-align: center; color:#4dffa6;'><----</h3>", unsafe_allow_html = True)
if st.sidebar.button('Submit'):
        data = {'gender': value(sex, male),
                'age': age,
                'weight': weight,
                'waist_circumference': waist_circumference,
                'systolic_bp': systolic_bp,
                'BMI': BMI,
                'hypertension': value(yn, hypertension),
                'take_HTN_medicine': value(yn, hypertension),
                'high_cholesterol': value(yn, high_cholesterol),
                'take_HCL_medicine': value(yn, take_HCL_medicine),
                'diabetes': value(yn, diabetes),
                'stroke': value(yn, stroke),
                'heart_failure': value(yn, heart_failure),
                'coronary_heart_disease': value(yn, coronary_heart_disease),
                'angina': value(yn, angina),
                'heart_attack': value(yn, heart_attack)}
#'heartRate': heart_rate,
#'glucose': glucose}
#'education': value(edu, education),
#'currentSmoker': value(yn, current_smoker),
#'cigsPerDay': cigsPerDay,  
        features = pd.DataFrame(data, index=[0])

        st.markdown("<h2 style='text-align: center; color:#000066;'>Data gathered........</h2>", unsafe_allow_html = True)
        st.markdown("<h2 style='text-align: center; color:#000066;'>Processing Results........</h2>", unsafe_allow_html = True)
        # Reads in saved classification model
        load_clf = pickle.load(open('stroke.pkl', 'rb'))
        
        # Apply model to make predictions
        prediction = load_clf.predict(features)
        prediction_proba = load_clf.predict_proba(features).reshape(2,)
        yes = prediction_proba[1]
        no = prediction_proba[0]
        

        
        
        st.markdown("<h2 style='text-align: center; color:#99ffff;'><u>Prediction </u></h2>", unsafe_allow_html = True)
        pred1, pred2, pred3 = st.beta_columns([12, 6, 14])
        if prediction==0:
            st.markdown("<h1 style='text-align: center; color:#006600;'>You don't have any heart problem.</h1>", unsafe_allow_html = True)
            with pred1:
                st.write("")
            with pred2:
                st.image("smile_emo.png")
            with pred3:
                st.write("")
        else:
            st.markdown("<h1 style='text-align: center; color:#cc0000;'>Go to a doctor.You may have heart problems.</h1>", unsafe_allow_html = True)
            with pred1:
                st.write("")
            with pred2:
                st.image("amb.png")
            with pred3:
                st.write("")
        
            
        st.markdown("<h1 style='text-align: center; color:#99ffff;'><u>Prediction Probability</u></h1>", unsafe_allow_html = True)
        fig,ax=plt.subplots(figsize=(10,8))
        axes=plt.bar(['Chances of being healthy\n{} %'.format(no*100),'Chances of getting cardiac diseases\n{} %'.format(yes*100)], [no, yes])
        axes[0].set_color('g')
        axes[1].set_color('r')
        st.pyplot(fig)
