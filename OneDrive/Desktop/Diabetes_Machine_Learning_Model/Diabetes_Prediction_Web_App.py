# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 12:18:21 2025

@author: migue
"""

import numpy as np
import pickle
import streamlit as st
import sklearn

#This pickle.load method opens our file we must mention the file name and 'rb' as a parameter which means read binary.
loaded_model = pickle.load(open('C:/Users/migue/OneDrive/Desktop/Diabetes_Machine_Learning_Model/trained_model.sav', 'rb'))


#Creating a function for prediction.

def diabetes_prediction(input_data):
    #Lets make a prediction system to see if our patient has diabetes or not.

    #Now we must change this input data to a numpy array.
    input_data_as_numpy_array = np.asarray(input_data)
    #Reshape the array because we are predicting for one instance

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    #Since the model is trained in this case on 768 data points 
    #We must reshape the array the parameter (1,-1) will tell the model
    #We are only predicting the label for one piece of data or instance.


    prediction = loaded_model.predict(input_data_reshaped)


    if (prediction[0] == '0'):
        return"You're patient does not have diabetes"
    else:
        return"You're patient has diabetes"
        


def main ():
    
    #Giving a title to our webapp
    st.title('Diabetes prediction Web App')
    
    #Getting the input from the user.
    
    Pregrnancies = st.text_input('Number of Pregnancies.')
    BloodPressure = st.text_input('BLood Pressure Value.')
    Glucose = st.text_input('Blood Glucose level.')
    SkinThickness = st.text_input('The Measurement of Skin Thickness.')
    BMI = st.text_input('Body Mass Index Value.')
    Age = st.text_input('Age of the person.')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value.')
    
    #Code for prediction.
    diagnosis = ''
    
    #Creating a button for prediction.
    
    if st.button('Diabetes test Result.'):
        diagnosis = diabetes_prediction([Pregnancies, BloodPressure, Glucose, SkinThickness, 
                                         BMI, Age, DiabetesPedigreeFunction])
        
    st.success(diagnosis)
    
    
    
    #This expression is used when you only want a particular function to be ran when using a commandprompt.
    
    if __name__  == '__main__':
        main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    