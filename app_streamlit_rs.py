import streamlit as st
import pickle
import numpy as np
import pandas as pd

#load the model and dataframe
df = pd.read_csv("df.csv")
pipe = pickle.load(open("pipe.pkl", "rb"))

st.title("Laptop Recommendation Systems")

#Now we will take user input one by one as per our dataframe

#Ram
ram = st.selectbox("Ram(in GB)", [2,4,6,8,12,16,24,32,64])

#screen size
screen_size = st.number_input('Screen Size')

#cpu
cpu = st.selectbox('CPU',df['Cpu_brand'].unique())

#memory
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

#Prediction
if st.button('Predict Price'):

    query = np.array([ram,cpu,hdd,ssd])
    query = query.reshape(1, 12)
    prediction = str(int(np.exp(pipe.predict(query)[0])))
    st.title("The predicted price of this configuration is " + prediction)
