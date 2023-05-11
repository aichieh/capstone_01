# Import packages
import streamlit as st
import pandas as pd 
import plotly.express as px
import plotly.io as pio
from PIL import Image
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve


# Set the plotly template
pio.templates.default = "plotly_white"

# Setting the title of the tab and the favicon
st.set_page_config(page_title='Examining Hypertension Using Health Care Data', page_icon = ':rain_cloud:', layout = 'centered')

# Setting the title on the page with some styling
st.markdown("<h1 style='text-align: center'>Examining Depression Using Health Care Data</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)

# Read the data
df1 = pickle.load(open('stroke.pkl', 'rb'))

# Creating the container for the first plot
#with st.beta_expander('Stroke Prediction'):

# Creating a selectbox dropdown with the categorical features to choose from
#    cat_option = st.selectbox('Select a feature to examine', cat_cols, key='cat_cols1')

# The code to run the first plot
X = df1.drop([target], axis=1)
y = df1[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lg=LogisticRegression(max_iter = 30000)
lg.fit(X_train,y_train)

#Predict training set:
y_pred = lg.predict(X_test)

CM = confusion_matrix(y_test, y_pred)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

result=pd.DataFrame()
# Sensitivity, hit rate, recall, or true positive rate
result['TPR'] = [round(TP/(TP+FN),2)]
 # Specificity or true negative rate
result['TNR'] = [round(TN/(TN+FP),2) ]
        # Fall out or false positive rate
result['FPR'] = [round(FP/(FP+TN),2)]
         # False negative rate
result['FNR'] = [round(FN/(TP+FN),2)]

lg_probs = lg.predict_proba(X_test)
        # keep probabilities for the positive outcome only
lg_probs = lg_probs[:, 1]
result['AUC'] = [round(roc_auc_score(y_test, lg_probs),2)]

gbc_auc = roc_auc_score(y_test, lg_probs)
gbc_fpr, gbc_tpr, gbc_thresholds = roc_curve(y_test, lg_probs)
plt.plot(gbc_fpr, gbc_tpr, label='ROC Curve (AUC = %0.2f)' % (gbc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='No skill')   
plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='green', label='Perfect Classifier')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc="lower right")
plt.show()
