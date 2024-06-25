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
    mentality_composure = st.slider('mentality_composure', 0, 100)
    mentality_vision = st.slider('mentality_vision', 0, 100)
    power_shot_power = st.slider('power_shot_power', 0, 100)
    movement_reactions = st.slider('movement_reactions', 0, 100)
    skill_ball_control = st.slider('skill_ball_control', 0, 100)
    skill_long_passing = st.slider('skill_long_passinge', 0, 100)
    age = st.number_input('age', min_value=15, max_value=60)
    attacking_short_passing = st.slider('attacking_short_passing', 0, 100)
    potential = st.slider('potential', 0, 100)

 
 
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
