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
        "predicted_cycle_length": pred_panjang,
        "predicted_period_length": pred_durasi,
        "predicted_peak_day": pred_hari_max,
        "predicted_dates": {
            "start_date": tanggal_mulai.strftime('%Y-%m-%d'),
            "end_date": tanggal_akhir.strftime('%Y-%m-%d'),
            "peak_date": tanggal_max.strftime('%Y-%m-%d')
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

@app.route('/add-cycle/<user_id>', methods=['POST'])
def add_cycle(user_id):
    path = os.path.join(DATA_ROOT, user_id)
    csv_path = os.path.join(path, "model.csv")
    os.makedirs(path, exist_ok=True)

    # Ambil data dari body request
    data = request.get_json()
    required_fields = ['start_date', 'durasi', 'hari_max_volume', 'panjang_siklus']

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Field '{field}' is required"}), 400

    # Baca file CSV kalau sudah ada
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        siklus_ke = df['siklus_ke'].max() + 1
    else:
        df = pd.DataFrame(columns=["user_id", "siklus_ke", "start_date", "durasi", "hari_max_volume", "panjang_siklus"])
        siklus_ke = 1

    # Tambahkan baris baru
    new_row = {
        "user_id": user_id,
        "siklus_ke": siklus_ke,
        "start_date": data["start_date"],
        "durasi": int(data["durasi"]),
        "hari_max_volume": int(data["hari_max_volume"]),
        "panjang_siklus": int(data["panjang_siklus"])
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)

    return jsonify({
    "message": "Cycle data added successfully",
    "siklus_ke": int(siklus_ke)
}), 200



# =========================
# 🏁 Run Server
# =========================
if __name__ == '__main__':
    os.makedirs(DATA_ROOT, exist_ok=True)
    app.run(debug=True)
