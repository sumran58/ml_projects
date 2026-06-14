import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#loading the saved models
diabetes_model=pickle.load(open('saved_models/trained_model.sav','rb'))

parkinsons_model=pickle.load(open('saved_models/parkinsons_model.sav','rb'))

with st.sidebar:
    selected=option_menu('Multiple Disease Prediction System',
                         ['Diabetes Prediction',
                          'Parkinsons Prediction'],default_index=0)
    
if(selected=='Diabetes Prediction'):
    st.title("diabetes Prediction using ML")

    Pregnancies=st.text_input("Number of pregnancies")
    Glucose=st.text_input("level of glucose")
    BloodPressure=st.text_input("level of BloodPressure")
    SkinThickness=st.text_input("skin thickness value")
    Insulin=st.text_input("Insulin value")
    BMI =st.text_input("Bmi level value")
    DiabetesPedigreeFunction=st.text_input("diabetes pedigree value")
    Age=st.text_input("Age of the person")

    diab_diagnosis=''
    if st.button("Diabetes Test Result"):

   
        diab_prediction = diabetes_model.predict([[
            float(Pregnancies),
            float(Glucose),
            float(BloodPressure),
            float(SkinThickness),
            float(Insulin),
            float(BMI),
            float(DiabetesPedigreeFunction),
            float(Age)
        ]])

        if diab_prediction[0] == 0:
            st.success("The person is non-diabetic")
        else:
            st.error("The person is diabetic")
    else:
        st.warning("Please enter all values")