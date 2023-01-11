import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder


st.sidebar.title('Car Price Prediction')
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML Cloud App </h2>
<p style="text-align: center;"><img src="https://i12.haber7.net//fotogaleri/haber7/album/2021/03/mercedes_volkswagen_bmw_audi_opel_skoda_toyota_renault_peugeot_fiat_seat_ve_honda_faizleri_dusurdu_1611393774_2581_w750_h450.jpg" class="img-fluid" width="600" height="350" alt="Auto"></p>
</div>"""


st.markdown(html_temp, unsafe_allow_html=True)


age=st.sidebar.selectbox("What is the age of your car:",(0,1,2,3))
hp=st.sidebar.slider("What is the hp_kw of your car?", 40, 300, step=5)
km=st.sidebar.slider("What is the km of your car", 0,350000, step=1000)
gearing_type=st.sidebar.radio('Select gear type',('Automatic','Manual','Semi-automatic'))
car_model=st.sidebar.selectbox("Select model of your car", ('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'))


qmodel=pickle.load(open("rf_model_new","rb"))
qtransformer = pickle.load(open('transformer', 'rb'))


my_dict = {
    "age": age,
    "hp_kW": hp,
    "km": km,
    'Gearing_Type':gearing_type,
    "make_model": car_model
    
}

df = pd.DataFrame.from_dict([my_dict])


st.header("The configuration of your car is below")
st.table(df)

df2 = qtransformer.transform(df)

st.subheader("Press predict if configuration is okay")

if st.button("Predict"):
    prediction = qmodel.predict(df2)
    #st.success("The estimated price of your car is €{}. ".format(int(prediction[0])))
    st.success("The estimated price of your car is €{}. ".format(prediction[0]))
