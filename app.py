import streamlit as st
import joblib
import numpy as np

file = "fifa_overall_rating_predictor.pkl"



# Load the trained model and scaler
ensemble_model = joblib.load(file)
scaler = joblib.load('scaler.joblib')

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

    return ensemble_model.predict(processed_input)[0]


def main():
    st.title("FIFA Player Overall Rating Prediction")

    # User input for features
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(feature.capitalize().replace('_', ' '), min_value=0, max_value=100, value=50)

    # Predict button
    if st.button("Predict Rating"):
        # Prepare the feature vector
        feature_values = np.array([list(input_data.values())])
        feature_values = scaler.transform(feature_values)
        
        # Make prediction
        prediction = ensemble_model.predict(feature_values)[0]
        
        # Display the prediction
        st.success(f"The predicted player overall rating is: {prediction:.2f}")

if __name__ == '__main__':
    main()
