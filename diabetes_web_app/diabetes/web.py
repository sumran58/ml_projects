import pandas as pd
import numpy as np
import streamlit as st
import pickle
loaded_model=pickle.load(open('trained_model.sav','rb'))

#creating a function for prediction
def diabetes_prediction(input_data):
    
    #change th type of input_data to numpy array
    input_data_as_nparray=np.asarray(input_data)
    #reshaping the array for predicting the one instance because we have trained our model on 768 data points
    input_data_reshaped=input_data_as_nparray.reshape(1,-1)
    #standardizing the data

    prediction=loaded_model.predict(input_data_reshaped)

    if prediction[0]==0:
        return("the person is non-diabetic")
    else:
        return("the person is diabetic")
    
def main():
    #giving a title
    st.title("Daibetes Prediction web app")
    Pregnancies=st.text_input("Number of Pregnancies")
    Glucose=st.text_input("Glucose level")
    BloodPressure=st.text_input("Blood Pressure value")
    SkinThickness=st.text_input("Skin Thickness  value")
    Insulin=st.text_input("Insulin Level")
    BMI=st.text_input("BMI value")
    DiabetesPedigreeFunction=st.text_input("Diabetes Pedigree Function Value")
    Age=st.text_input("Age of the Person")


    diagnosis=''
    if st.button('Diabetes test result'):
        diagnosis = diabetes_prediction([
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ])

    st.success(diagnosis)

if __name__=='__main__':
    main()


        
    