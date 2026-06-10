import numpy as np
import pandas as pd
import pickle
#loading the saved model from the notebook 
loaded_model=pickle.load(open('trained_model.sav','rb'))
#making a predictive system
input_data=(1	,89	,66	,23	,94,	28.1,	0.167	,21)
#change th type of input_data to numpy array
input_data_as_nparray=np.asarray(input_data)
#reshaping the array for predicting the one instance because we have trained our model on 768 data points
input_data_reshaped=input_data_as_nparray.reshape(1,-1)
#standardizing the data

prediction=loaded_model.predict(input_data_reshaped)

if prediction[0]==0:
  print("the person is non-diabetic")
else:
  print("the person is diabetic")

