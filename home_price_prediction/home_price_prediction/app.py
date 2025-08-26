import joblib
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model
model = joblib.load('home_price_model_new123.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]

    # Predict using the model
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted Home Price: $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
