from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np

# create a Flask web application
app = Flask(__name__)

# load the trained model from a pickle file
model = pickle.load(open('linear_regression_model.pkl', 'rb'))

# define a route to serve the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# define a route to accept POST requests with input data
@app.route('/predict', methods=['POST'])
def predict():
    # get the input data from the request form
    max_temp = float(request.form['max_temp'])
    min_temp = float(request.form['min_temp'])
    apparent_temp_max = float(request.form['apparent_temp_max'])
    apparent_temp_min = float(request.form['apparent_temp_min'])
    shortwave_radiation = float(request.form['shortwave_radiation'])
    evaporation = float(request.form['evaporation'])
    windspeed_max = float(request.form['windspeed_max'])
    winddirection_dominant = float(request.form['winddirection_dominant'])

    # preprocess the input data as necessary
    input_array = np.array([
        max_temp,
        min_temp,
        apparent_temp_max,
        apparent_temp_min,
        shortwave_radiation,
        evaporation,
        windspeed_max,
        winddirection_dominant
    ]).reshape(1, -1)

    # make a prediction using the loaded model
    prediction = model.predict(input_array)

    # return the prediction as a JSON response
    return render_template('index.html', prediction_text='The predicted value is {:.2f}'.format(prediction[0]))


# start the Flask web server
if __name__ == '__main__':
    app.run(debug=True, port='8081')
