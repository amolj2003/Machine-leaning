from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model
loaded_model = joblib.load('linear_regression_modell.pkl')

# Define the home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        crim = float(request.form['crim'])
        zn = float(request.form['zn'])
        indus = float(request.form['indus'])
        chas = float(request.form['chas'])
        rm = float(request.form['rm'])
        age = float(request.form['age'])
        dis = float(request.form['dis'])
        rad = float(request.form['rad'])
        tax = float(request.form['tax'])
        ptratio = float(request.form['ptratio'])
        b = float(request.form['b'])
        lstat = float(request.form['lstat'])

        # Create a numpy array from form data
        sample = np.array([[crim, zn, indus, chas, rm, age, dis, rad, tax, ptratio, b, lstat]])

        # Make prediction using the loaded model
        predicted_value = loaded_model.predict(sample)[0]

        return render_template('index.html', prediction=predicted_value,calculation=10)
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return render_template('index.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
