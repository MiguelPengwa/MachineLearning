import pickle
import streamlit as st 
import numpy as np 


#Loading the saved model.

#This pickle.load method opens our file we must mention the file name and 'rb' as a parameter which means read binary.
loaded_model = pickle.load(open('C:/Users/migue/OneDrive/Desktop/Diabetes_Prediction_Web_App/trained_model.sav', 'rb'))


def diabetes_prediction(input_data):
    #Lets make a prediction system to see if our patient has diabetes or not.

    input_data = (0,137,40,35,168,43.1,2.288,3)
    #Now we must change this input data to a numpy array.
    input_data_as_numpy_array = np.asarray(input_data)
    #Reshape the array because we are predicting for one instance

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    #Since the model is trained in this case on 768 data points 
    #We must reshape the array the parameter (1,-1) will tell the model
    #We are only predicting the label for one piece of data or instancel.

    #Standardize the data we use the std function

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == '0'):
        print("You're patient does not have diabetes")
    else:
        print("You're patient has diabetes")




def main():
    
    st.title('The Diabetes Prediction Web App')
    
    #Getting the input from our user.
    
    Pregnancies = st.text_input('The number of Pregnancies')
    Glucose = st.text_input('The value of Glucose level')
    BloodPressure = st.text_input('The value of the Blood Pressure')
    SkinThickness = st.text_input('The value of the Skin Thickness')
    Insulin = st.text_input('The value of the Insulin level')
    Bmi = st.text_input('The value of the Bmi')
    DiabetesPedigreeFunction = st.text_input('The value of the Diabetes Pedigree Function')
    Age = st.text_input('The person Age')
    
    diagnosis = ''
    
    if st.button('The diabetes test results'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure,
                                         SkinThickness, Insulin, Bmi, DiabetesPedigreeFunction, Age])
        
        st.success(diagnosis)
        
        
if __name__ == '__main__':
    main()
        
        