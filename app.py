import streamlit as st
import pickle
import numpy as np

# Load model
model, scaler = pickle.load(open("titanic_model.pkl", "rb"))

st.title("Titanic Survival Prediction")

Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1,2,3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 600.0, 50.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode inputs
sex_val = 1 if Sex == "male" else 0
embarked_map = {"C":0, "Q":1, "S":2}
embarked_val = embarked_map[Embarked]

features = np.array([[Pclass, sex_val, Age, SibSp, Parch, Fare, embarked_val]])
features = scaler.transform(features)

if st.button("Predict"):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.success("🎉 This passenger would have SURVIVED!")
    else:
        st.error("❌ This passenger would NOT have survived.")
