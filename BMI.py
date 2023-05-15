import streamlit as st
import plotly.graph_objects as go

def calculate_bmi(weight, height):
    bmi = weight / (height ** 2)
    return bmi

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

# Streamlit App
st.title("BMI Indicator Gauges")

weight = st.number_input("Enter your weight (in kg):", min_value=0.0)
height = st.number_input("Enter your height (in meters):", min_value=0.0)

if st.button("Submit"):
    bmi = calculate_bmi(weight, height)
    st.write("Your BMI:", bmi)
    #st.plotly_chart(fig, use_container_width=True)
    # Generate a grid of BMI gauges

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

