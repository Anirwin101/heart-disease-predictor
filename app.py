from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained ML model
model = pickle.load(open("heart-model.pkl", "rb"))

# Feature order MUST match training
FEATURES = [
    'age', 'gender', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal'
]

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')

    try:
        input_data = []
        user_name = request.form.get('name', 'User')

        # Collect and type-cast input
        for feature in FEATURES:
            if feature in ['gender', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
                value = int(request.form[feature])
            else:
                value = float(request.form[feature])
            input_data.append(value)

        input_array = np.array(input_data).reshape(1, -1)

        # Prediction
        prediction = model.predict(input_array)[0]

        # Probabilities
        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_array)[0]

            # Always show disease probability
            disease_index = list(model.classes_).index(0)
            probability = round(proba[disease_index] * 100, 2)

        # Final result label
        result = "High Risk" if prediction == 0 else "Low Risk"

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
