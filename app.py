import streamlit as st
import joblib
import pandas as pd
import numpy as np

file = "fifa_overall_rating_predictor.pkl"



# Load the trained model and scaler
ensemble_model = joblib.load(file)

# Define features (Replace these with your actual top 13 feature names)
features = ['potential',
 'age',
 'attacking_short_passing',
 'skill_long_passing',
 'skill_ball_control',
 'movement_reactions',
 'power_shot_power',
 'mentality_vision',
 'mentality_composure']


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
    mentality_composure = st.number_input('mentality_composure', min_value=0, max_value=100)
    mentality_vision = st.number_input('mentality_vision', min_value=0, max_value=100)
    power_shot_power = st.number_input('power_shot_power', min_value=0, max_value=100)
    movement_reactions = st.number_input('movement_reactions', min_value=0, max_value=100)
    skill_ball_control = st.number_input('skill_ball_control', min_value=0, max_value=100)
    skill_long_passing = st.number_input('skill_long_passinge', min_value=0, max_value=100)
    age = st.number_input('age', min_value=15, max_value=60)
    attacking_short_passing = st.number_input('attacking_short_passing', min_value=0, max_value=100)
    potential = st.number_input('potential', min_value=0, max_value=100)

 
 
    #input_data = {}
    #for feature in features:
        #input_data[feature] = st.number_input(feature.capitalize().replace('_', ' '), min_value=0, max_value=100, value=50)

    # Predict button
    if st.button("Predict Rating"):
        # Prepare the feature vector
        input_data = {
         'mentality_composure' : mentality_composure,
         'mentality_vision' : mentality_vision,
         'power_shot_power' : power_shot_power,
         'movement_reactions' : movement_reactions,
         'skill_ball_control' : skill_ball_control,
         'skill_long_passing' : skill_long_passing,
         'age': age,
         'attacking_short_passing' : attacking_short_passing,
         'potential' : potential,
        }
        
        # Make prediction
        prediction = predict_player_rating(input_data)
        
        # Display the prediction
        st.success(f"The predicted player overall rating is: {prediction:.2f}")

if __name__ == '__main__':
    main()
