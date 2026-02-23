"""
TITANIC SURVIVAL PREDICTION - STREAMLIT APP
============================================

A web application to predict passenger survival on the Titanic
using a trained Logistic Regression model.

To run this app:
    streamlit run titanic_streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .survived {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .died {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        with open('titanic_logreg_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('titanic_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('titanic_feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please ensure all pickle files are in the same directory.")
        return None, None, None

model, scaler, feature_names = load_model()

# Title and description
st.markdown('<h1 class="main-header">🚢 Titanic Survival Prediction</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
    Predict whether a passenger would have survived the Titanic disaster using Machine Learning
</div>
""", unsafe_allow_html=True)

# Sidebar - About
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/300px-RMS_Titanic_3.jpg", 
             use_column_width=True)
    
    st.markdown("## About This App")
    st.markdown("""
    This application uses a **Logistic Regression** model trained on the famous Titanic dataset 
    to predict passenger survival.
    
    ### Model Performance
    - **Accuracy**: 84.36%
    - **ROC-AUC**: 0.8701
    - **Precision**: 0.8021
    - **Recall**: 0.7500
    
    ### Key Factors
    1. Gender (female = higher survival)
    2. Passenger Class (1st > 2nd > 3rd)
    3. Age (children favored)
    4. Fare (wealth indicator)
    5. Family Size
    
    ---
    
    ### Technologies Used
    - Python
    - Scikit-learn
    - Streamlit
    - Pandas
    
    ---
    
    Made with ❤️ for Data Science
    """)

# Main content tabs
tab1, tab2, tab3 = st.tabs(["🎯 Make Prediction", "📊 Model Insights", "ℹ️ Help"])

with tab1:
    st.markdown('<h2 class="sub-header">Enter Passenger Information</h2>', unsafe_allow_html=True)
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Personal Details")
        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            format_func=lambda x: f"{'1st' if x==1 else '2nd' if x==2 else '3rd'} Class",
            help="Ticket class: 1st (Upper), 2nd (Middle), 3rd (Lower)"
        )
        
        sex = st.radio(
            "Gender",
            options=["Female", "Male"],
            horizontal=True
        )
        
        age = st.slider(
            "Age",
            min_value=0,
            max_value=80,
            value=30,
            help="Passenger age in years"
        )
        
        name = st.text_input(
            "Passenger Name (optional)",
            placeholder="e.g., Smith, Mr. John"
        )
    
    with col2:
        st.markdown("### Family Information")
        sibsp = st.number_input(
            "Siblings/Spouses Aboard",
            min_value=0,
            max_value=10,
            value=0,
            help="Number of siblings or spouses traveling with the passenger"
        )
        
        parch = st.number_input(
            "Parents/Children Aboard",
            min_value=0,
            max_value=10,
            value=0,
            help="Number of parents or children traveling with the passenger"
        )
        
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        
        st.info(f"Total Family Size: **{family_size}**")
        st.info(f"Traveling Alone: **{'Yes' if is_alone else 'No'}**")
    
    with col3:
        st.markdown("### Travel Details")
        fare = st.number_input(
            "Ticket Fare (£)",
            min_value=0.0,
            max_value=500.0,
            value=32.0,
            step=1.0,
            help="Price paid for the ticket in pounds"
        )
        
        embarked = st.selectbox(
            "Port of Embarkation",
            options=["Southampton", "Cherbourg", "Queenstown"],
            help="Where the passenger boarded the ship"
        )
        
        cabin = st.checkbox(
            "Had Cabin Number",
            value=False,
            help="Whether the passenger had a cabin number recorded"
        )
    
    # Determine title based on age and gender
    if name:
        if "Mr." in name or "Mr " in name:
            title = "Mr"
        elif "Mrs." in name or "Mrs " in name:
            title = "Mrs"
        elif "Miss" in name or "Ms" in name:
            title = "Miss"
        elif "Master" in name:
            title = "Master"
        else:
            title = "Rare"
    else:
        if sex == "Male":
            title = "Master" if age < 18 else "Mr"
        else:
            title = "Miss" if age < 30 else "Mrs"
    
    # Determine age band
    if age <= 12:
        age_band = "Child"
    elif age <= 18:
        age_band = "Teen"
    elif age <= 35:
        age_band = "Adult"
    elif age <= 60:
        age_band = "Middle"
    else:
        age_band = "Senior"
    
    # Determine fare band
    if fare < 7.91:
        fare_band = "Low"
    elif fare < 14.45:
        fare_band = "Medium"
    elif fare < 31.0:
        fare_band = "High"
    else:
        fare_band = "VeryHigh"
    
    # Map embarked port
    embarked_map = {"Southampton": "S", "Cherbourg": "C", "Queenstown": "Q"}
    embarked_code = embarked_map[embarked]
    
    # Predict button
    st.markdown("---")
    predict_button = st.button("🔮 Predict Survival", type="primary", use_container_width=True)
    
    if predict_button and model is not None:
        # Prepare input features
        input_dict = {
            'Pclass': pclass,
            'Sex': 1 if sex == "Male" else 0,  # Male=1, Female=0
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'FamilySize': family_size,
            'IsAlone': is_alone,
            'HasCabin': 1 if cabin else 0
        }
        
        # Add one-hot encoded features
        for port in ['C', 'Q', 'S']:
            input_dict[f'Embarked_{port}'] = 1 if embarked_code == port else 0
        
        for t in ['Master', 'Miss', 'Mr', 'Mrs', 'Rare']:
            input_dict[f'Title_{t}'] = 1 if title == t else 0
        
        for ab in ['Adult', 'Child', 'Middle', 'Senior', 'Teen']:
            input_dict[f'AgeBand_{ab}'] = 1 if age_band == ab else 0
        
        for fb in ['High', 'Low', 'Medium', 'VeryHigh']:
            input_dict[f'FareBand_{fb}'] = 1 if fare_band == fb else 0
        
        # Create DataFrame with all required features
        input_df = pd.DataFrame([input_dict])
        
        # Ensure all features are present (add missing ones as 0)
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training
        input_df = input_df[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Prediction Result</h2>', unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box survived">
                <h2 style='text-align: center; color: #28a745;'>✅ SURVIVED</h2>
                <p style='text-align: center; font-size: 1.3rem;'>
                    This passenger would likely have <strong>SURVIVED</strong> the Titanic disaster
                </p>
                <p style='text-align: center; font-size: 1.1rem;'>
                    Survival Probability: <strong>{probability[1]*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box died">
                <h2 style='text-align: center; color: #dc3545;'>❌ DID NOT SURVIVE</h2>
                <p style='text-align: center; font-size: 1.3rem;'>
                    This passenger would likely have <strong>NOT SURVIVED</strong> the Titanic disaster
                </p>
                <p style='text-align: center; font-size: 1.1rem;'>
                    Survival Probability: <strong>{probability[1]*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show probability breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(['Did Not Survive', 'Survived'], probability, color=['#dc3545', '#28a745'], alpha=0.7)
            ax.set_ylabel('Probability', fontweight='bold')
            ax.set_title('Survival Probability Breakdown', fontweight='bold')
            ax.set_ylim(0, 1)
            for i, p in enumerate(probability):
                ax.text(i, p + 0.02, f'{p*100:.1f}%', ha='center', fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("### Key Factors")
            factors = []
            
            if sex == "Female":
                factors.append("✓ Female (strong positive factor)")
            else:
                factors.append("✗ Male (strong negative factor)")
            
            if pclass == 1:
                factors.append("✓ 1st Class ticket (positive)")
            elif pclass == 3:
                factors.append("✗ 3rd Class ticket (negative)")
            
            if age < 18:
                factors.append("✓ Child (positive)")
            
            if fare > 50:
                factors.append("✓ High fare paid (positive)")
            elif fare < 10:
                factors.append("✗ Low fare paid (negative)")
            
            if family_size > 1 and family_size < 5:
                factors.append("✓ Moderate family size (positive)")
            elif family_size > 4:
                factors.append("✗ Large family (negative)")
            
            if is_alone:
                factors.append("⚠ Traveling alone (mixed effect)")
            
            for factor in factors:
                st.markdown(f"- {factor}")

with tab2:
    st.markdown('<h2 class="sub-header">Model Performance & Insights</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Score': [0.8436, 0.8021, 0.7500, 0.7753, 0.8701]
        })
        
        for _, row in metrics_df.iterrows():
            st.metric(row['Metric'], f"{row['Score']:.4f}")
    
    with col2:
        st.markdown("### Top Survival Factors")
        st.markdown("""
        Based on feature importance analysis:
        
        1. **Gender** (Female = +3.24 odds multiplier)
        2. **Passenger Class** (1st class = +2.1x)
        3. **Title** (Mrs/Miss = positive)
        4. **Age** (Children favored)
        5. **Fare** (Higher = better)
        6. **Family Size** (Moderate is best)
        
        ### Historical Context
        - Overall survival rate: **38.4%**
        - Female survival rate: **74.2%**
        - Male survival rate: **18.9%**
        - 1st class survival: **63.0%**
        - 3rd class survival: **24.2%**
        """)
    
    st.markdown("---")
    st.markdown("### Model Understanding")
    
    st.markdown("""
    **Logistic Regression** calculates survival probability using this formula:
    
    ```
    P(Survival) = 1 / (1 + e^-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ))
    ```
    
    Where:
    - β₀ = Intercept
    - βᵢ = Coefficient for feature i
    - Xᵢ = Feature value
    
    **Key Insights:**
    - Positive coefficients increase survival probability
    - Negative coefficients decrease survival probability
    - The model combines all factors to make a prediction
    """)

with tab3:
    st.markdown('<h2 class="sub-header">Help & Information</h2>', unsafe_allow_html=True)
    
    with st.expander("📖 How to Use This App"):
        st.markdown("""
        1. **Enter passenger details** in the input fields on the "Make Prediction" tab
        2. Click the **"Predict Survival"** button
        3. View the prediction and probability breakdown
        4. Explore the key factors influencing the prediction
        
        All fields are required for accurate prediction.
        """)
    
    with st.expander("🎓 About the Titanic Dataset"):
        st.markdown("""
        The Titanic dataset contains information about passengers on the ill-fated maiden voyage 
        of the RMS Titanic in 1912.
        
        **Dataset Features:**
        - Passenger Class (1st, 2nd, 3rd)
        - Gender and Age
        - Family members aboard
        - Ticket fare
        - Port of embarkation
        - Survival outcome
        
        **Historical Facts:**
        - Departed: April 10, 1912 from Southampton
        - Sank: April 15, 1912 at 2:20 AM
        - Passengers: ~2,224
        - Survivors: ~710 (32%)
        - "Women and children first" policy was enforced
        """)
    
    with st.expander("🤖 About Logistic Regression"):
        st.markdown("""
        Logistic Regression is a statistical method for **binary classification** 
        (predicting one of two outcomes).
        
        **Why it works well for Titanic:**
        - Clear binary outcome (survived/died)
        - Interpretable coefficients
        - Fast predictions
        - Handles mixed feature types well
        
        **Model Training:**
        - Trained on 712 passengers
        - Validated on 179 passengers
        - 26 engineered features
        - Achieved 84.36% accuracy
        """)
    
    with st.expander("❓ FAQs"):
        st.markdown("""
        **Q: How accurate is this prediction?**
        A: The model achieves 84.36% accuracy on validation data. However, remember 
        it's a statistical prediction based on historical patterns.
        
        **Q: Why is gender so important?**
        A: The "women and children first" evacuation protocol during the Titanic 
        disaster led to much higher survival rates for females (74% vs 19% for males).
        
        **Q: Does passenger class really matter?**
        A: Yes! 1st class passengers had 63% survival rate vs 24% for 3rd class. 
        This was due to cabin location and access to lifeboats.
        
        **Q: Can I save my predictions?**
        A: Currently, this demo doesn't save predictions. You can take a screenshot 
        of the results.
        
        **Q: What data was used to train this model?**
        A: The famous Kaggle Titanic dataset with 891 passenger records.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Titanic Survival Predictor v1.0 | Built with Streamlit & Scikit-learn</p>
    <p>⚠️ This is a machine learning model for educational purposes. 
    Historical outcome predictions should not be taken as definitive.</p>
</div>
""", unsafe_allow_html=True)
