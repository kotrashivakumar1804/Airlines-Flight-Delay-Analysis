from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('flight_delay_regressor.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        origin = request.form['originAirport']
        destination = request.form['destinationAirport']
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hour = int(request.form['hour'])
        minute = int(request.form['minute'])
        flight_number = int(request.form['flightNumber'])
        airline = request.form['airline']
        tail_number = request.form['tailNumber']
        distance = int(request.form['distance'])

        # Prepare input for model (only these fields)
        input_data = {
            'YEAR': [year],
            'MONTH': [month],
            'DAY': [day],
            'FLIGHT_NUMBER': [flight_number],
            'TAIL_NUMBER': [tail_number],
            'AIRLINE': [airline],
            'ORIGIN_AIRPORT': [origin],
            'DESTINATION_AIRPORT': [destination],
            'DISTANCE': [distance],
            'SCHED_DEP_HOUR': [hour],
            'SCHED_DEP_MIN': [minute]
        }
        input_df = pd.DataFrame(input_data)

        # Encode categorical columns
        for col, le in label_encoders.items():
            if col in input_df.columns and input_df[col].dtype == 'object':
                input_df[col] = le.transform(input_df[col].astype(str))

        # Ensure all columns are present and in the correct order
        for col in model.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model.feature_names_in_]

        # Predict arrival delay
        predicted_delay = float(model.predict(input_df)[0])

        # Convert delay to hours and minutes
        delay_hours = int(predicted_delay // 60)
        delay_minutes = int(predicted_delay % 60)

        # Delay status logic
        if predicted_delay < 0:
            delay_status = "Early Arrival"
        elif predicted_delay == 0:
            delay_status = "On Time"
        elif predicted_delay <= 15:
            delay_status = "Short Delay"
        elif predicted_delay <= 45:
            delay_status = "Moderate Delay"
        else:
            delay_status = "Significant Delay"

        # Render result
        return render_template(
            'result.html',
            origin_airport=origin,
            destination_airport=destination,
            distance=distance,
            delay_hours=delay_hours,
            delay_minutes=delay_minutes,
            delay_status=delay_status
        )

    except Exception as e:
        print("Prediction error:", e)
        return render_template('error.html', error=f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=8080)
    