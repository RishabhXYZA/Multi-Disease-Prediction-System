import streamlit as st
import joblib 
import numpy as np
import pandas as pd
st.set_page_config(
    page_title="Multi Disease Predictor",
    page_icon="🩺",  
    layout="wide"
)

def set_background_url(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )



#### Sidebar ####
st.sidebar.header("⚖️ Multi-Disease Predictor")
st.sidebar.markdown("\n\n")
st.sidebar.header("🗂️ Dashboard Menu")
st.sidebar.subheader("Please select the Disease you are concerned about.")
st.sidebar.markdown("\n\n")
st.sidebar.markdown("\n\n")
st.sidebar.markdown("\n\n")

#### ---- ####
#### INTRODUCTION ####

page = st.sidebar.selectbox("Choose from Below 👇", ["🌐 Home", "💉 Diabetes", "🫁 Kidney","🫀 Heart Health","📊 Conclusions"])

if page == "🌐 Home":
    st.title("Multi-Disease Predictor Prediction App")
    st.header("Welcome to the Multi-Disease Prediction System")
    st.write("""The Multi-Disease Prediction System is a smart healthcare application that uses 
             Machine Learning models to analyze user health data and predict the likelihood of major diseases 
             like Diabetes, Kidney Disease, and Heart Disease. It aims to promote early detection and awareness 
             for better health decisions.""")
    
    col1, col2, col3 = st.columns(3)
 
    with col1: 
        st.subheader("Diabetes")
        st.image("Images/2029e90127a71935f75689863af49fefbaeb66f0227b2add6bed3dda.png", use_container_width=True)
        with st.expander("Know more"):
          st.write("""Diabetes is a chronic metabolic disorder that occurs when the body is unable to properly 
                   regulate blood glucose (sugar) levels. It results either from insufficient insulin production by the 
                   pancreas or the body's inability to effectively use the insulin it produces. 
                   There are several types of diabetes, with Type 1 and Type 2 being the most common. 
                   If not managed properly, diabetes can lead to serious long-term health complications 
                   affecting the heart, kidneys, eyes, and nerves. While there is no permanent cure, 
                   it can be effectively managed through lifestyle changes, medication, and regular monitoring of 
                   blood sugar levels.""")
        with st.expander("Symptoms"):
            st.subheader('Key Symptoms')
            st.write("* Frequent urination (polyuria)")
            st.write("* Excessive thirst (polydipsia)")
            st.write("* Increased hunger (polyphagia)")
            st.write("* Fatigue or feeling tired")
            st.write("* Blurred vision")
            st.write("* Slow healing of cuts and wounds")
            st.write("* Frequent infections (e.g., skin, gum, or bladder infections)")
            st.write("* Dry mouth and itchy skin")  

    with col2:
        st.subheader("Kidney")
        st.image("Images/images.jpeg", use_container_width=True)
        with st.expander("Know more"):
          st.write("""Kidney disease occurs when the kidneys are unable to efficiently filter waste products, 
                   excess fluids, and toxins from the blood. This leads to an accumulation of harmful substances in the body, 
                   which can cause serious complications such as high blood pressure, electrolyte imbalances, 
                   and swelling in various parts of the body. Kidney disease can develop gradually (chronic kidney disease) 
                   or suddenly (acute kidney injury), and its progression may go unnoticed in the early stages. 
                   Common causes include diabetes, hypertension, infections, genetic factors, and prolonged use of 
                   certain medications. If not properly managed, kidney disease can lead to kidney failure, requiring dialysis 
                   or transplantation. While there is no complete cure for chronic kidney disease, early detection, 
                   lifestyle modifications, proper medication, and regular monitoring can slow its progression and 
                   improve quality of life.""")
        with st.expander("Symptoms"):
            st.write("* Swelling in feet, ankles, or hands (edema)")
            st.write("* Fatigue and weakness")
            st.write("* Difficulty concentrating")
            st.write("* Changes in urination (frequency, color, or pain)")
            st.write("* Persistent itching")
            st.write("* Shortness of breath")
            st.write("* Nausea or vomiting")
            st.write("* Muscle cramps")  

    with col3:
        st.subheader("Heart Health")
        st.image("Images/a72f5f26f9f6daf42a9c96021359010121e8f6e0cee81d0e756990d7.jpg",use_container_width=True)
        with st.expander("Know more"):
            st.write("""Heart health is essential for the proper functioning of the body, 
                     as the heart is responsible for pumping oxygen-rich blood to all organs and tissues. 
                     Maintaining good cardiovascular health supports energy levels, mental clarity, and the body’s 
                     ability to recover from illness or stress. It involves a combination of healthy lifestyle choices, 
                     such as eating nutrient-rich foods, staying physically active, managing stress, and 
                     getting regular medical check-ups. Factors like genetics, environment, 
                     and daily habits all influence heart health over time. Fostering heart health not only 
                     helps prevent life-threatening conditions but also contributes to a longer, more active, 
                     and fulfilling life.""")
        with st.expander("Symptoms"):
            st.subheader('Key Symptoms')
            st.write("* Chest pain, tightness, or discomfort")
            st.write("* Shortness of breath, especially during activity or while lying down")
            st.write("* Irregular heartbeat (palpitations or fluttering)")
            st.write("* Cold sweats or nausea")
            st.write("* Fainting or near-fainting episodes")
            st.write("* Reduced ability to exercise or perform physical activities")
            st.write("* Swelling in legs, ankles, or feet (edema)")
            st.write("* Pain in the neck, jaw, throat, upper back, or arms")  

    # ------------------- Dataset Download Section -------------------
        
    st.markdown("---")
    st.markdown("## 📥 Download Datasets")

    # Function to safely load CSV files
    def load_csv(file_path):
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            st.warning(f"File not found: {file_path}")
            return None

    # Load datasets safely
    diabetes_df = load_csv("Datasets/diabetes.csv")
    kidney_df = load_csv("Datasets/kidney_disease.csv")
    heart_df = load_csv("Datasets/heart.csv")

    colA, colB, colC = st.columns(3)

    # Download buttons only if data loaded successfully
    if diabetes_df is not None:
        with colA:
            st.subheader("Diabetes Dataset")
            st.download_button(
                label="Click here to download Diabetes Dataset",
                data=diabetes_df.to_csv(index=False),
                file_name="diabetes_dataset.csv",
                mime="text/csv"
            )

    if kidney_df is not None:
        with colB:
            st.subheader("Kidney Dataset")
            st.download_button(
                label="Click here to download Kidney Dataset",
                data=kidney_df.to_csv(index=False),
                file_name="kidney_dataset.csv",
                mime="text/csv"
            )

    if heart_df is not None:
        with colC:
            st.subheader("Heart Dataset")
            st.download_button(
                label="Click here to download Heart Dataset",
                data=heart_df.to_csv(index=False),
                file_name="heart_dataset.csv",
                mime="text/csv"
            )


elif page == "🫁 Kidney":

    st.title("🫁 Kidney's Predictor ")
    st.subheader("Please fill the values below from your Report")

    with st.form("in"):
        st.subheader("Input Values : ")

        features = []
        feature_names = [
            "age","bp","al (Albumin)","su (Sugar)","rbc ","pc (Pus Cells)","pcc (Pus cell Clumps)",
            "ba (Bacteria)","bgr (Blood Glucose Random)","bu (Blood Urea)","sc (Serum Creatinine)","pot (Potassium)",
            "wc (White Blood Cell Count)","htn (Hypertension)","dm (Diabetes Mellitius)",
            "cad (Coronary Artery disease)","pe (Pedal Edema)","ane (Anemia)"
        ] 

        for name in feature_names:
            val = st.number_input(name, min_value=-100.0, max_value=200.0000, value=0.00, step=0.1,format='%.6g')
            features.append(val)

        submitted = st.form_submit_button("Submit")

        if submitted:
          model_kidney = joblib.load("Kidney.joblib")

          # Inside your form submission
          prediction = model_kidney.predict([features])


          if prediction[0] == 1:
            st.markdown(
              "<div style='color: white; background-color: red; padding: 8px; border-radius: 5px;'>"
              "⚠️ <strong>Kidney’s Disease Detected</strong>, Please contact a Doctor"
              "</div>",
                unsafe_allow_html=True
            )
          else:
            st.markdown(
              "<div style='color: white; background-color: green; padding: 8px; border-radius: 5px;'>"
              "✅ <strong>No Kidney’s Disease</strong>, but if you have symptoms, Please confirm with a Doctor"
              "</div>",
              unsafe_allow_html=True
            )

elif page == "💉 Diabetes":
    st.title("💉 Diabetes Predictor")
    st.subheader("Please fill the values below from your Report")
    

    with st.form("in"):
        st.subheader("Input Values : ")

        features1 = []
        feature_names1 = [
            'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'
        ] 

        for name in feature_names1:
            val = st.number_input(name, min_value=0.0, max_value=541.0, value=0.00, step=1.0,format='%.3f')
            features1.append(val)

        submitted = st.form_submit_button("Submit")

        if submitted:
          scaler = joblib.load("diabetes_scaler_IT.joblib")
          model_diabetes = joblib.load("diabetes_IT.joblib")

          # Inside your form submission
          features_scaled = scaler.transform([features1])
          prediction1 = model_diabetes.predict(features_scaled)


          if prediction1[0] == 1:
            st.markdown(
              "<div style='color: white; background-color: red; padding: 8px; border-radius: 5px;'>"
              "⚠️ <strong>You have Diabetes.</strong> Please contact a Doctor"
              "</div>",
                unsafe_allow_html=True
            )
          else:
            st.markdown(
              "<div style='color: white; background-color: green; padding: 8px; border-radius: 5px;'>"
              "✅ <strong>No Diabetes detected</strong>, but if you have symptoms, Please confirm with a Doctor"
              "</div>",
              unsafe_allow_html=True
            )

elif page == "🫀 Heart Health":
    st.title(" Heart Health Predictor")
    st.subheader("Please fill the values below from your Report")
    with st.form("in"):
        st.subheader("Input Values : ")

        features2 = []
        feature_names2 = [
            'age','sex (0-> Feamle, 1->Male)','cp (Chest Pain Type)','trestbps (Resting BP)','chol (Cholestrol)',
            'fbs (Fasting Blood Sugar)','restecg (Resting Electrocardiographic)',
            'thalach (Maximum Heart Rate)','exang (Exercise-Induced Angina)',
            'oldpeak (ST Depression)','slope (ST Segment Slope)','ca (Number of Coronary Arteries)','thal (Thalassemia)'
        ]  

        for name in feature_names2:
            val = st.number_input(name, min_value=0.0, max_value=500.0, value=0.00, step=1.0,format='%.1f')
            features2.append(val)

        submitted = st.form_submit_button("Submit")
        if submitted:
          scaler = joblib.load("heart_scaler.joblib")
          model_heart = joblib.load("heart.joblib")

          # Inside your form submission
          features_scaled = scaler.transform([features2])
          prediction1 = model_heart.predict(features_scaled)


          if prediction1[0] == 1:
            st.markdown(
              "<div style='color: white; background-color: red; padding: 8px; border-radius: 5px;'>"
              "⚠️ <strong>Your Heart Health is poor.</strong> Please contact a Doctor"
              "</div>",
                unsafe_allow_html=True
            )
          else:
            st.markdown(
              "<div style='color: white; background-color: green; padding: 8px; border-radius: 5px;'>"
              "💓<strong>Healthy Heart Detected</strong>, but if you have symptoms, Please confirm with a Doctor"
              "</div>",
              unsafe_allow_html=True
            )

elif page == "📊 Conclusions":
    
    st.title("📊 Conclusions & Model Analysis")
    st.markdown("Review the performance of each disease prediction model below:")

    tab1, tab2, tab3 = st.tabs([
        "🫁 Kidney",
        "💉 Diabetes",
        "🫀 Heart Disease"
    ])

    with tab1:
        st.header("🫁 Kidney Disease")

        with st.expander("🔍 Model Details"):
          st.markdown("""
        ### 🧪 **Model Summary**
        - ✅ **Model Used:** Random Forest Classifier  
        - 🧠 **Features Used:**
            - `age`
            - `bp`
            - `al`
            - `su`
            - `rbc`
            - `pc`
            - `pcc`
            - `ba`
            - `bgr`
            - `bu`
            - `sc`
            - `pot`
            - `wc`
            - `htn`
            - `dm`
            - `cad`
            - `pe`
            - `ane`
        """)



        with st.expander("📈 Performance Metrics"):
            st.markdown("""
        ### ✅ **Performance Overview**
        - **Accuracy:** `93.75%`
        - **Precision:**  
          - ❌ No Kidney's Disease: `0.90`  
          - ✅ Kidney's Disease: `0.96`
        - **Recall:**  
          - ❌ No Kidney's Disease: `0.93`  
          - ✅ Kidney's Disease: `0.94`
        - **F1 Score:**  
          - ❌ No Kidney's Disease: `0.91`  
          - ✅ Kidney's Disease: `0.95`
        """)

        with st.expander("📊 Confusion Matrix and Visuals"):
            st.subheader("Factors affecting for Positive Kidney's and Negative Kidney Disease")
            st.image('Images/subplot_kidney.png')
            st.subheader("Heatmap for kidney disease")
            st.image('Images/kidney_heatmap.png',caption="Heatmap")
            st.subheader("Confusion Matrix")
            st.image("Images/cm_kidney.png", caption="Confusion Matrix")

        with st.expander("🧠 Observations & Insights"):
            st.markdown("""
            - High accuracy achieved due to clear separation of features such as blood urea, serum creatinine, and age.
            - Random Forest captures non-linear relationships and handles missing or noisy data effectively.
            - Feature importance is highest for serum creatinine, blood urea, and hypertension history.
            - Model demonstrates good sensitivity in detecting early-stage kidney disease.
            - Ensemble methods like Random Forest reduce overfitting.
            """)

    with tab2:
        st.header("💉 Diabetes Prediction")
        with st.expander("🔍 Model Details"):
          st.markdown("""
        ### 🧪 **Model Summary**
        - ✅ **Model Used:** Random Forest Classifier 
        - 📏 **Scaler:** StandardScaler  
        - 🧠 **Features Used:**
            - `Pregnancies`
            - `Glucose`
            - `BloodPressure`
            - `SkinThickness`
            - `Insulin`
            - `BMI`
            - `DiabetesPedigreeFunction`
            - `Age`""",unsafe_allow_html=True
          )



        with st.expander("📈 Performance Metrics"):
           st.markdown("""
        ### ✅ **Performance Overview**
        - **Accuracy:** `79.2%`
        - **Precision:**  
          - 🟩 **No Diabetes (Class 0):** `0.79`  
          - 🟥 **Diabetes (Class 1):** `0.81`
        - **Recall:**  
          - 🟩 **No Diabetes (Class 0):** `0.93`  
          - 🟥 **Diabetes (Class 1):** `0.55`
        - **F1 Score:**  
          - 🟩 **No Diabetes (Class 0):** `0.85`  
          - 🟥 **Diabetes (Class 1):** `0.65`
        """)

        with st.expander("📊 Confusion Matrix and Visuals"):
            st.subheader("Factors affecting the Outcome for Diabetes")
            st.image('Images/subplot_diabetes.png',caption="Factors affecting the Outcome for Diabetes")
            st.subheader("Heatmap for Diabetes")
            st.image("Images/heatmap_diabetes.png", caption="Heatmap")
            st.subheader("Confusion Matrix")
            st.image("Images/cm_diabetes.png", caption="Confusion Matrix")

        with st.expander("🧠 Observations & Insights"):
          st.markdown("""
        - The model is good at identifying **non-diabetic individuals**.
        - It **misses some diabetic cases** (low recall for Class 1).
        - Suggests a need for better **recall sensitivity** toward diabetic predictions.

        """)



    with tab3:
        st.header("🫀 Heart Disease Prediction")
        with st.expander("🔍 Model Details"):
          st.markdown("""
        ### 🧪 **Model Summary**
        - **✅ Model Used:Random Forest Classifier**  
        - **📏 Scaler:** StandardScaler  
        - **🧬 Encoding:** No Encoding  
        - **🧠 Features Used:**  
          - `Age`
          - `Sex`
          - `CP`
          - `Trestbps`
          - `Cholesterol`
          - `FastingBS`
          - `RestingECG`
          - `Thalach`
          - `ExerciseAngina`
          - `Oldpeak`
          - `ST_Slope`
          - `CA` 
          - `Thal`          
        """)

        with st.expander("📈 Model Performance Summary"):
          st.markdown("""
        ### ✅ **Performance Overview**
        - **Accuracy:** `83.52%`
        - **Precision:**  
          - 💚 **No Heart Disease (0):** `0.84`  
          - ❤️ **Heart Disease (1):** `0.83`
        - **Recall:**  
          - 💚 **No Heart Disease (0):** `0.78`  
          - ❤️ **Heart Disease (1):** `0.88`
        - **F1 Score:**  
          - 💚 **No Heart Disease (0):** `0.81`  
          - ❤️ **Heart Disease (1):** `0.85`
        """)

        with st.expander("📊 Confusion Matrix and Visuals"):
            st.header("*Important Features Impacting Prediction*")
            st.subheader("Features that affects target")
            st.image("Images/subplot_heart.png", caption="Features affecting heart health the most")
            st.subheader("Heatmap for Heart Disease")
            st.image("Images/heart_heatmap.png", caption="Heatmap")
            st.subheader("Confusion Matrix")
            st.image("Images/cm_heart.png", caption="Confusion Matrix")

        with st.expander("🧠 Observations & Insights"):
          st.markdown("""
        - **High accuracy** (83.52%) tells model is well-trained.
        -  Model performs **very well for detecting non-heart disease cases** (High F1-Score).
        -  **Precision for heart disease is very good**, so when the model predicts it, it's mostly correct.
        """)


#### ---- ####
st.sidebar.markdown("\n\n")
st.sidebar.markdown("---")
st.sidebar.markdown("\n\n")
st.sidebar.markdown("\n\n")
st.sidebar.markdown("\n\n")
st.sidebar.markdown("\n\n")
st.sidebar.markdown("\n\n")
st.sidebar.markdown("\n\n")
st.sidebar.markdown("\n\n")
st.sidebar.markdown("\n\n")

st.sidebar.subheader("Thank you for choosing MediPredict❤️— your intelligent partner for early disease detection.")

#### ---- ####
