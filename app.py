import streamlit as st
import joblib
import pandas as pd



try:
    model = joblib.load("fifa_overall_rating_predictor.pkl")
    st.success("Success!")
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.stop()


common_features = ['skill_ball_control', 
                   'mentality_vision', 
                   'power_shot_power', 
                   'potential', 
                   'movement_reactions', 
                   'mentality_composure', 
                   'age', 
                   'attacking_short_passing', 
                   'skill_long_passing'
]


def preprocess_info(data):
    input_df = pd.DataFrame([data])
    input_df = input_df[common_features]
    return input_df


def predict_rating(data):
    processed_input = preprocess_info(data)
    prediction = model.predict(processed_input)
    return prediction[0]

st.title('FIFA Player Rating Predictor')


potential = st.slider('Potential', 0, 100)
age = st.number_input('Age', min_value=15, max_value=50)
attacking_short_passing = st.slider('Short Passing', 0, 100)
skill_long_passing = st.slider('Long Passing', 0, 100)
skill_ball_control = st.slider('Ball Control', 0, 100)
movement_reactions = st.slider('Movement Reactions', 0, 100)
power_shot_power = st.slider('Power Shot', 0, 100)
mentality_vision = st.slider('Mentality Vision', 0, 100)
mentality_composure = st.slider('Mentality Composure', 0, 100)



if st.button('Predict Rating'):
    data = {
        'potential': potential,
        'age': age,
        'attacking_short_passing': attacking_short_passing,
        'skill_long_passing': skill_long_passing,
        'skill_ball_control': skill_ball_control,
        'movement_reactions': movement_reactions,
        'power_shot_power': power_shot_power,
        'mentality_vision': mentality_vision,
        'mentality_composure': mentality_composure,
        
    }
    
    
    try:
        prediction = predict_rating(data)
        st.success(f'Predicted player rating: {prediction:.2f}')
    except ValueError as e:
        st.error(f"Prediction error: {str(e)}")
