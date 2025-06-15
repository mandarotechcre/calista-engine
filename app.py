import os
import time
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime, timedelta
import joblib
import threading

app = Flask(__name__)
DATA_ROOT = "data"

# ========================
# Util: Model Loader Aman
# ========================
def safe_load_model(path, retries=3, delay=0.5):
    for attempt in range(retries):
        try:
            return joblib.load(path)
        except (OSError, EOFError):
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

# ========================
# Train Model per User
# ========================
def train_user_model(user_id):
    path = os.path.join(DATA_ROOT, user_id)
    csv_path = os.path.join(path, "model.csv")
    model_path = os.path.join(path, "model.pkl")
    tmp_model_path = model_path + ".tmp"

    if not os.path.exists(csv_path):
        print(f"❌ Data not found for {user_id}")
        return None

    df = pd.read_csv(csv_path)
    if "user_id" in df.columns:
        df = df.drop(columns=["user_id"])
    df = df.sort_values(by='siklus_ke')

    if len(df) < 2:
        print(f"❌ Not enough data to train for {user_id}")
        return None

    X, Y = [], []
    records = df.to_dict('records')
    for i in range(len(records) - 1):
        curr = records[i]
        next_ = records[i + 1]
        X.append([curr['durasi'], curr['hari_max_volume'], curr['panjang_siklus']])
        Y.append([next_['panjang_siklus'], next_['hari_max_volume'], next_['durasi']])

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X, Y)

    os.makedirs(path, exist_ok=True)
    joblib.dump(model, tmp_model_path)
    os.replace(tmp_model_path, model_path)

    print(f"✅ Model trained for {user_id}")
    return model

# ========================
# Endpoint: Predict
# ========================
@app.route('/predict/<user_id>', methods=['POST'])
def predict(user_id):
    path = os.path.join(DATA_ROOT, user_id)
    csv_path = os.path.join(path, "model.csv")
    model_path = os.path.join(path, "model.pkl")

    if not os.path.exists(csv_path):
        return jsonify({"error": "CSV data not found"}), 400

    df = pd.read_csv(csv_path)
    if "user_id" in df.columns:
        df = df.drop(columns=["user_id"])
    df = df.sort_values(by='siklus_ke')

    if len(df) < 2:
        return jsonify({"error": "Not enough data to predict"}), 400

    last = df.iloc[-1]
    input_features = [[last['durasi'], last['hari_max_volume'], last['panjang_siklus']]]

    try:
        if os.path.exists(model_path):
            model = safe_load_model(model_path)
        else:
            model = train_user_model(user_id)
            if model is None:
                return jsonify({"error": "Training failed"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

    pred = model.predict(input_features)[0]
    pred_panjang = round(pred[0])
    pred_max = round(pred[1])
    pred_durasi = round(pred[2])

    start = datetime.strptime(last['start_date'], "%Y-%m-%d")
    tgl_mulai = start + timedelta(days=pred_panjang)
    tgl_akhir = tgl_mulai + timedelta(days=pred_durasi - 1)
    tgl_max = tgl_mulai + timedelta(days=pred_max - 1)

    return jsonify({
        "predicted_cycle_length": pred_panjang,
        "predicted_period_length": pred_durasi,
        "predicted_peak_day": pred_max,
        "predicted_dates": {
            "start_date": tgl_mulai.strftime('%Y-%m-%d'),
            "end_date": tgl_akhir.strftime('%Y-%m-%d'),
            "peak_date": tgl_max.strftime('%Y-%m-%d')
        }
    })

# ========================
# Endpoint: Train
# ========================
@app.route('/train/<user_id>', methods=['POST'])
def train_api(user_id):
    def async_train():
        train_user_model(user_id)

    thread = threading.Thread(target=async_train)
    thread.start()
    return jsonify({"message": f"Training started for {user_id}."})

# ========================
# Endpoint: Add Cycle
# ========================
@app.route('/add-cycle/<user_id>', methods=['POST'])
def add_cycle(user_id):
    path = os.path.join(DATA_ROOT, user_id)
    csv_path = os.path.join(path, "model.csv")
    os.makedirs(path, exist_ok=True)

    data = request.get_json()
    required = ['start_date', 'durasi', 'hari_max_volume', 'panjang_siklus']
    for field in required:
        if field not in data:
            return jsonify({"error": f"{field} is required"}), 400

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "user_id" in df.columns:
            df = df.drop(columns=["user_id"])
        siklus_ke = df['siklus_ke'].max() + 1
    else:
        df = pd.DataFrame(columns=["siklus_ke", "start_date", "durasi", "hari_max_volume", "panjang_siklus"])
        siklus_ke = 1

    new_row = {
        "siklus_ke": siklus_ke,
        "start_date": data["start_date"],
        "durasi": int(data["durasi"]),
        "hari_max_volume": int(data["hari_max_volume"]),
        "panjang_siklus": int(data["panjang_siklus"])
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)

    return jsonify({"message": "Cycle added", "siklus_ke": int(siklus_ke)})

# ========================
# Run
# ========================
if __name__ == '__main__':
    os.makedirs(DATA_ROOT, exist_ok=True)
    app.run(debug=True)
