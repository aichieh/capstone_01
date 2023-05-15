import streamlit as st
import plotly.graph_objects as go

def create_bmi_gauge(bmi_value):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = bmi_value,
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
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
    return fig

# BMI Meter
st.title("BMI Meter")
def calculate_bmi(weight, height):
    weight_kg = weight*2.205
    height_m = (height*2.54)/100
    bmi = weight_kg / (height_m ** 2)
    return bmi

check = st.sidebar.button('Submit')
if(check):
    bmi = calculate_bmi(weight, height) 
    st.title(f'Your BMI : {bmi}')
    if bmi <18.5:
        st.title("You are Underweight")
    elif bmi>= 18.5 and bmi<25:
        st.title("You are Normal")
    elif bmi >=25 and bmi<30:
        st.title("You are Overweight")
    else:
        st.title("You are Obese")

# Streamlit App
st.title("BMI Indicator Gauges")
       
#if st.button("Submit"):
#    bmi = calculate_bmi(weight, height)
#    st.write("Your BMI:", bmi)
    #st.plotly_chart(fig, use_container_width=True)
    # Generate a grid of BMI gauges

    # Update layout and display the plot
    fig.update_layout(height=600, width=800, title="BMI Indicators")
    fig.show()

