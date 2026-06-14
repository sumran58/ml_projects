from fastapi import FastAPI
from pydantic import BaseModel
import json
import pickle

app=FastAPI()
class model_input(BaseModel):
    Pregnancies : int
    Glucose :int
    BloodPressure :int
    SkinThickness :int
    Insulin : int
    BMI :float
    DiabetesPedigreeFunction :float
    Age:int 

diabetes_model=pickle.load(open('trained_model.sav','rb'))

@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters: model_input):
    input_data=input_parameters.json() #sending the data to the model in the form of json 
    input_dictionary=json.loads(input_data) #convertingt the json format data in dictionary 
    preg=input_dictionary['Pregnancies']
    glu=input_dictionary['Glucose']
    BP=input_dictionary['BloodPressure']
    ST=input_dictionary['SkinThickness']
    insu=input_dictionary['Insulin']
    bmi=input_dictionary['BMI']
    DPF=input_dictionary['DiabetesPedigreeFunction']
    age=input_dictionary['Age']

    input_list=[preg,glu,BP,ST,insu,bmi,DPF,age]

    prediction=diabetes_model.predict([input_list])
    if prediction[0]==0:
        return 'The person is non-diabetic'
    else:
        return 'The person is Diabetic'
