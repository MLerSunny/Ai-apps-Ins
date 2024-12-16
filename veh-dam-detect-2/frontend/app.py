import streamlit as st
import requests

st.title("🚗 Vehicle Damage Detection")
st.write("Upload an image to check if the vehicle is damaged.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/predict/", files=files)
    prediction = response.json()["Prediction"]

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Prediction:** {prediction}")