import streamlit as st
import joblib
import numpy as np
import pandas as pd

file = "fifa_overall_rating_predictor.pkl"

# Load the trained model
ensemble_model = joblib.load(file)

# Define features
features = [
    'potential', 'age', 'shooting', 'passing', 'dribbling', 'physic',
    'attacking_short_passing', 'skill_long_passing', 'skill_ball_control',
    'movement_reactions', 'power_shot_power', 'mentality_vision', 'mentality_composure'
]

# Define preprocessing function
def preprocess_input_data(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[features]
    return input_df

# Define prediction function
def predict_player_rating(input_data):
    processed_input = preprocess_input_data(input_data)
    predicted = ensemble_model.predict(processed_input)
    return predicted[0]

def main():
    st.title("FIFA Player Overall Rating Prediction")

    # User input for features
    potential = st.slider('potential', 0, 100)
    age = st.number_input('age', min_value=15, max_value=60)
    shooting = st.slider('shooting', 0, 100)
    passing = st.slider('passing', 0, 100)
    dribbling = st.slider('dribbling', 0, 100)
    physic = st.slider('physic', 0, 100)
    attacking_short_passing = st.slider('attacking_short_passing', 0, 100)
    skill_long_passing = st.slider('skill_long_passing', 0, 100)
    skill_ball_control = st.slider('skill_ball_control', 0, 100)
    movement_reactions = st.slider('movement_reactions', 0, 100)
    power_shot_power = st.slider('power_shot_power', 0, 100)
    mentality_vision = st.slider('mentality_vision', 0, 100)
    mentality_composure = st.slider('mentality_composure', 0, 100)

    # Predict button
    if st.button("Predict Rating"):
        # Prepare the feature vector
        input_data = {
            'potential': potential,
            'age': age,
            'shooting': shooting,
            'passing': passing,
            'dribbling': dribbling,
            'physic': physic,
            'attacking_short_passing': attacking_short_passing,
            'skill_long_passing': skill_long_passing,
            'skill_ball_control': skill_ball_control,
            'movement_reactions': movement_reactions,
            'power_shot_power': power_shot_power,
            'mentality_vision': mentality_vision,
            'mentality_composure': mentality_composure
        }

        # Make prediction
        prediction = predict_player_rating(input_data)

        # Display the prediction
        st.success(f"The predicted player overall rating is: {prediction:.2f}")

if __name__ == '__main__':
    main()
