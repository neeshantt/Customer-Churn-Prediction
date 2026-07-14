# gender --> female = 1, male =0
# churn --> yes = 1, no = 0
#scaler is imported as scaler.pkl
# model is exported as model.pkl
#order of x--> "Age","Gender","Tenure","MonthlyCharges"



import streamlit as st
import pandas as pd
import pickle
import numpy as np


scaler = pickle.load(open('scaler.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.title("CHURN PREDICTION")
st.divider()

st.write("Please enter the following details for prediction")

st.divider()

age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
gender = st.selectbox("Select Gender", options=["Male", "Female"])
tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)
monthly_charges = st.number_input("Enter Monthly Charges", min_value=30, max_value=150, value=50)

st.divider()

predictbutton = st.button("predict")

if predictbutton:
   gender = 1 if gender == "Female" else 0

   x = [[age, gender, tenure, monthly_charges]]
   x1 = np.array(x)
   x_scaled = scaler.transform(x1)
   prediction = model.predict(x_scaled)[0]

   predicted = "Churn/Yes" if prediction == 1 else "Not Churn/No"
   st.write(f"The predicted result is: {predicted}")


else:
   
   st.write("Please enter the details and click on predict button to get the prediction")
