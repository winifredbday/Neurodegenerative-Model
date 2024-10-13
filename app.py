from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

# Load the trained model and symptom columns
model = joblib.load('disease_prediction_model.pkl')
symptom_columns = joblib.load('symptom_columns.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    # List of symptoms (You can later make this dynamic, e.g., fetching from a database)
    symptoms = [
        "memory_loss", "confusion", "difficulty_recognizing_people", 
        "poor_judgment", "cognitive_decline", "tremors", "slowness", 
        "stooped_posture", "small_handwriting", "drooling", 
        "muscle_weakness", "speech_difficulty", "difficulty_swallowing", 
        "muscle_cramps", "difficulty_holding_objects", "chorea", 
        "mood_swings", "seizures", "personality_changes"
    ]
    
    # Render the HTML form for inputting symptoms
    return render_template('symptom_form.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected symptoms from the form (checkboxes)
    selected_symptoms = request.form.getlist('symptoms')  # 'symptoms' corresponds to the name of the checkboxes
    
    # Predict the disease based on the selected symptoms
    prediction = predict_disease(selected_symptoms)
    
    # Render the result on the webpage
    return render_template('result.html', predicted_disease=prediction)

def predict_disease(symptoms):
    # Create a dictionary with all symptoms initialized to 0
    symptom_dict = {col: 0 for col in symptom_columns}
    
    # Set the symptoms selected by the user to 1
    for symptom in symptoms:
        if symptom in symptom_dict:
            symptom_dict[symptom] = 1
    
    # Convert to DataFrame for the model
    input_data = pd.DataFrame([symptom_dict])
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    return prediction[0]

if __name__ == '__main__':
    app.run(debug=True)
