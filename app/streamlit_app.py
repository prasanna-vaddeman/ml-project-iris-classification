import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model
model = joblib.load("app/iris_model.pkl")


# Image files
flower_images = {
    0: "app/images/iris_setosa.jpg",
    1: "app/images/iris_versicolor.jpg",
    2: "app/images/iris_virginica.jpg"
}

flower_names = ["Setosa", "Versicolor", "Virginica"]


# -------------------- HEADER --------------------
st.markdown(
    """
    <div style='background-color:#0d6efd; padding:20px; border-radius:10px;'>
        <h1 style='text-align: center; color: white;'>Iris Flower Classification App</h1>
        <p style='text-align: center; color: #f8f9fa; font-size: 18px;'>
            Predict the Iris flower type using Sepal and Petal measurements.
        </p>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)



# -------------------- INPUT SECTION --------------------
st.markdown("### Input Features")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.4)
    sepal_width  = st.slider("Sepal width (cm)",  2.0, 4.5, 3.4)

with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.5)
    petal_width  = st.slider("Petal width (cm)",  0.1, 2.5, 0.2)

predict_button = st.button("Predict Flower Type", use_container_width=True)


# -------------------- PREDICTION --------------------
if predict_button:

    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    flower_name = flower_names[pred]

    # SUCCESS BOX
    st.markdown(
        f"""
        <div style="
            background-color:#0f5132;
            padding:18px;
            border-radius:10px;
            color:white;
            font-size:22px;
            text-align:center;
            margin-top:20px;
        ">
            Predicted Flower: <b>{flower_name}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show image with border + shadow
    st.markdown("<br>", unsafe_allow_html=True)
    st.image(
        flower_images[pred],
        width=350,
        caption=flower_name,
    )

    # ---------------- PROBABILITY BAR CHART ----------------
    st.markdown("### 📊 Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Flower": flower_names,
        "Probability": probs
    })

    st.bar_chart(prob_df.set_index("Flower"))
