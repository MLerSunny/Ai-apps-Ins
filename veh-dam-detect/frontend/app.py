import streamlit as st
import requests

st.title("🚗 Vehicle Damage Detection")
model_type = st.selectbox("Select Model:", ["CNN", "Mask R-CNN"])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(
        "http://127.0.0.1:8000/predict/",
        files=files,
        data={"model_type": model_type.lower()},
    )
    prediction = response.json()

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Model Used:** {prediction['model']}")
    st.write(f"**Prediction:** {prediction['prediction']}")
