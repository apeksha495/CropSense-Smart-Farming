from flask import Flask, render_template, request, session, redirect
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'crop_sense_secret_key'
  # Language translation session ke liye

# ✅ Load Models and Data
model = joblib.load('gaytriiiii.pkl')
label_encoders = joblib.load('label_encoders (1).pkl')
price_df = pd.read_csv('agmarknet.csv')

ALLOWED_SOIL_TYPES = list(label_encoders['soil_type'].classes_)
ALLOWED_CROP_TYPES = list(label_encoders['crop_type'].classes_)

# Dummy Translation Function (agar aap Flask-Babel use kar rahe hain toh _() automatically kaam karega)
def _(text):
    return text

@app.route('/')
def home():
    # Language switch handling
    lang = request.args.get('lang')
    if lang:
        session['lang'] = lang
    
    return render_template('index.html',
                           soil_types=ALLOWED_SOIL_TYPES,
                           crop_types=ALLOWED_CROP_TYPES,
                           SOIL_TRANSLATIONS={},
                           CROP_TRANSLATIONS={},
                           _ = _)
@app.route('/set_language', methods=['GET'])
def set_language():
    # URL se 'lang' parameter nikalein (e.g. 'hi' ya 'en')
    lang = request.args.get('lang', 'en')
    session['lang'] = lang  # Session me save karein
    
    # Request jahan se aayi thi (index/report page), wahan wapas redirect kar do
    return redirect(request.referrer or '/')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()

        original_soil_type = form_data['soil_type'].strip()
        original_crop_type = form_data['crop_type'].strip()

        # Inputs convert to float
        numeric_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'moisture']
        for field in numeric_fields:
            form_data[field] = float(form_data[field])

        # Encodings
        encoded_soil = label_encoders['soil_type'].transform([original_soil_type])[0]
        encoded_crop = label_encoders['crop_type'].transform([original_crop_type])[0]

        input_array = np.array([
            form_data['N'], form_data['P'], form_data['K'],
            form_data['temperature'], form_data['humidity'], form_data['ph'],
            form_data['rainfall'], form_data['moisture'],
            encoded_soil, encoded_crop
        ]).reshape(1, -1)

        # ML Prediction
        predicted_crop = model.predict(input_array)[0]

        # 💰 1. Market Price Prediction (agmarknet.csv se)
        crop_price_data = price_df[price_df['Commodity'].str.lower() == str(predicted_crop).lower()]
        if not crop_price_data.empty:
            avg_price = crop_price_data['Modal Price'].mean()
            price_text = f"Estimated Market Price: ₹ {avg_price:.2f} per Quintal"
        else:
            price_text = "Market price data not available for this crop."

        # 📋 2. Detailed Report
        report_text = (
            f"Crop: {predicted_crop}\n"
            f"Soil Type: {original_soil_type}\n"
            f"N-P-K Ratio: {form_data['N']} - {form_data['P']} - {form_data['K']}\n"
            f"Temperature: {form_data['temperature']}°C | Humidity: {form_data['humidity']}%\n"
            f"pH: {form_data['ph']} | Rainfall: {form_data['rainfall']} mm"
        )

        # 🤖 3. AI Explanation
        gpt_text = (
            f"{predicted_crop.capitalize()} is recommended based on the provided soil nutrients "
            f"and environmental conditions like temperature ({form_data['temperature']}°C) "
            f"and rainfall ({form_data['rainfall']}mm)."
        )

        prediction_text = f"🌾 Recommended Crop: {predicted_crop}"

        # 🚀 Output ko 'report.html' par bhej rahe hain
        return render_template(
            'report.html',
            prediction_text=prediction_text,
            price_text=price_text,
            report_text=report_text,
            gpt_text=gpt_text,
            _ = _
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"❌ Error: {str(e)}",
            soil_types=ALLOWED_SOIL_TYPES,
            crop_types=ALLOWED_CROP_TYPES,
            SOIL_TRANSLATIONS={},
            CROP_TRANSLATIONS={},
            _ = _
        )

if __name__ == "__main__":
    app.run(debug=True)