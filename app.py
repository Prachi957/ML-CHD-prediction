from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('chd_model.pkl')  # Ensure model accepts 14 features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        education = int(request.form['education'])
        sex = int(request.form['sex'])
        is_smoking = int(request.form['is_smoking'])
        cigsPerDay = float(request.form['cigsPerDay'])
        BPMeds = int(request.form['BPMeds'])
        prevalentStroke = int(request.form['prevalentStroke'])
        prevalentHyp = int(request.form['prevalentHyp'])
        diabetes = int(request.form['diabetes'])
        totChol = float(request.form['totChol'])
        sysBP = float(request.form['sysBP'])
        diaBP = float(request.form['diaBP'])
        BMI = float(request.form['BMI'])
        heartRate = float(request.form['heartRate'])
        glucose = float(request.form['glucose'])

        # Derived feature
        pulsepressure = sysBP - diaBP

        features = np.array([[age, education, sex, is_smoking, cigsPerDay,
                              BPMeds, prevalentStroke, prevalentHyp, diabetes,
                              totChol, BMI, heartRate, glucose, pulsepressure]])

        prediction = model.predict(features)[0]
        result = "High CHD Risk (1)" if prediction == 1 else "Low CHD Risk (0)"

        return render_template('form.html', prediction=result)
    except Exception as e:
        return render_template('form.html', prediction=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)


