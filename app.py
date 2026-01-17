from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained ML model
model = pickle.load(open("heart-model.pkl", "rb"))

# feature order must match training
FEATURES = [
    'age', 'gender', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal'
]

# Homepage route
@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/contact')
def anirwin():
    return render_template('contact.html')

# Predict route (handles both GET and POST)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Show the form
        return render_template('index.html')
    
    try:
        # Collect input from form
        input_data = []
        user_name = request.form.get('name', 'User')  

        
        for feature in FEATURES:
            if feature in ['gender','cp','fbs','restecg','exang','slope','ca','thal']:
                value = int(request.form[feature])
            else:
                value = float(request.form[feature])
            input_data.append(value)

        input_array = np.array(input_data).reshape(1, -1)        
        prediction = model.predict(input_array)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_array)[0]
            pred_index = list(model.classes_).index(prediction)
            probability = round(proba[pred_index]*100, 2)

        result = "High Risk" if prediction == 0 else "Low Risk"

        if result == "Low Risk" and probability <= 80:
            probability = -round((probability - 100), 2)  
        elif result == "High Risk" and probability >= 20:
            probability = probability



        # Render result page
        return render_template(
            "result.html",
            prediction=result,
            probability=probability,
            user_name=user_name
        )

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)