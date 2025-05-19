import os
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime, timedelta
import joblib
import threading

app = Flask(__name__)
DATA_ROOT = "data"

# =========================
# 🔁 Train Function
# =========================
def train_user_model(user_id):
    path = os.path.join(DATA_ROOT, user_id)
    csv_path = os.path.join(path, "model.csv")
    model_path = os.path.join(path, "model.pkl")

    if not os.path.exists(csv_path):
        print(f"❌ Data not found for user: {user_id}")
        return None

    df = pd.read_csv(csv_path)
    df = df.sort_values(by='siklus_ke')

    if len(df) < 2:
        print(f"❌ Not enough data to train model for {user_id}")
        return None

    X, Y = [], []
    rows = df.to_dict('records')
    for i in range(len(rows) - 1):
        curr, next_ = rows[i], rows[i + 1]
        X.append([curr['durasi'], curr['hari_max_volume'], curr['panjang_siklus']])
        Y.append([next_['panjang_siklus'], next_['hari_max_volume'], next_['durasi']])

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X, Y)
    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved for user {user_id}")
    return model

# =========================
# 🔮 Predict
# =========================
@app.route('/predict/<user_id>', methods=['POST'])
def predict(user_id):
    path = os.path.join(DATA_ROOT, user_id)
    csv_path = os.path.join(path, "model.csv")
    model_path = os.path.join(path, "model.pkl")

    if not os.path.exists(csv_path):
        return jsonify({"error": "CSV data not found"}), 400

    df = pd.read_csv(csv_path)
    df = df.sort_values(by='siklus_ke')

    if len(df) < 2:
        return jsonify({"error": "Not enough data to predict"}), 400

    last = df.iloc[-1]
    input_features = [[last['durasi'], last['hari_max_volume'], last['panjang_siklus']]]

    # Load or train model
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = train_user_model(user_id)
        if model is None:
            return jsonify({"error": "Training failed due to insufficient data"}), 400

    pred = model.predict(input_features)[0]
    pred_panjang = round(pred[0])
    pred_hari_max = round(pred[1])
    pred_durasi = round(pred[2])

    start_date = datetime.strptime(last['start_date'], "%Y-%m-%d")
    tanggal_mulai = start_date + timedelta(days=pred_panjang)
    tanggal_akhir = tanggal_mulai + timedelta(days=pred_durasi - 1)
    tanggal_max = tanggal_mulai + timedelta(days=pred_hari_max - 1)

    return jsonify({
        "📊 Prediksi Siklus Berikutnya": {
            "Panjang siklus": f"{pred_panjang} hari",
            "Hari maksimal volume darah": f"{pred_hari_max} hari",
            "Durasi haid": f"{pred_durasi} hari"
        },
        "📅 Prediksi Tanggal Penting Siklus Berikutnya": {
            "Tanggal mulai haid": tanggal_mulai.strftime('%Y-%m-%d'),
            "Tanggal akhir haid": tanggal_akhir.strftime('%Y-%m-%d'),
            "Tanggal darah terbanyak": tanggal_max.strftime('%Y-%m-%d')
        }
    })

# =========================
# 🔁 Async Training Endpoint (opsional)
# =========================
@app.route('/train/<user_id>', methods=['POST'])
def train_api(user_id):
    def async_train():
        train_user_model(user_id)

    thread = threading.Thread(target=async_train)
    thread.start()
    return jsonify({"message": f"Training started asynchronously for user {user_id}."})

# =========================
# 🏁 Run Server
# =========================
if __name__ == '__main__':
    os.makedirs(DATA_ROOT, exist_ok=True)
    app.run(debug=True)
