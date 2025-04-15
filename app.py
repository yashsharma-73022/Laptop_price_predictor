import streamlit as st
import pickle
import numpy as np
import base64

# Function to set animated background
def set_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set your animated background here
set_bg_from_local("Background.webp")

# Load model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("💻 Laptop Price Predictor")

# UI Elements
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

# Prediction
if st.button('Predict Price'):
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size

    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    predicted_price = int(np.exp(pipe.predict(query)[0]))
    st.title(f"💰 Predicted Price: ₹ {predicted_price}")
