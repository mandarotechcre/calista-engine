import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime, timedelta

# === 1. Load Data ===
df = pd.read_csv('menstrual_data.csv')
df = df.sort_values(by=['user_id', 'siklus_ke'])

# === 2. Siapkan Data Latih ===
X, Y = [], []
rows = df.to_dict('records')

for i in range(len(rows) - 1):
    curr = rows[i]
    next_ = rows[i + 1]
    if curr['user_id'] == next_['user_id']:
        # Input dari siklus sekarang
        X.append([
            curr['durasi'],
            curr['hari_max_volume'],
            curr['panjang_siklus']
        ])
        # Output dari siklus berikutnya
        Y.append([
            next_['panjang_siklus'],
            next_['hari_max_volume'],
            next_['durasi']
        ])

# Cek apakah cukup data
if len(X) < 1:
    print("❌ Tidak cukup data untuk melatih model.")
    exit()

# === 3. Latih Model ===
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X, Y)

# === 4. Prediksi Berdasarkan Siklus Terakhir ===
last = df[df['user_id'] == 1].sort_values(by='siklus_ke').iloc[-1]
new_input = [[last['durasi'], last['hari_max_volume'], last['panjang_siklus']]]
pred = model.predict(new_input)[0]

# Ambil hasil prediksi
pred_panjang_siklus = round(pred[0])
pred_hari_max = round(pred[1])
pred_durasi = round(pred[2])

# === 5. Hitung Tanggal Berdasarkan Start Date Terakhir ===
start_date_terakhir = datetime.strptime(last['start_date'], "%Y-%m-%d")

tanggal_mulai_haid = start_date_terakhir + timedelta(days=pred_panjang_siklus)
tanggal_akhir_haid = tanggal_mulai_haid + timedelta(days=pred_durasi - 1)
tanggal_max_volume = tanggal_mulai_haid + timedelta(days=pred_hari_max - 1)

# === 6. Tampilkan Hasil ===
print("\n📊 Prediksi Siklus Berikutnya:")
print(f"- Panjang siklus: {pred_panjang_siklus} hari")
print(f"- Hari maksimal volume darah: {pred_hari_max} hari")
print(f"- Durasi haid: {pred_durasi} hari")

print("\n📅 Prediksi Tanggal Penting Siklus Berikutnya:")
print(f"- Tanggal mulai haid: {tanggal_mulai_haid.strftime('%Y-%m-%d')}")
print(f"- Tanggal akhir haid: {tanggal_akhir_haid.strftime('%Y-%m-%d')}")
print(f"- Tanggal darah terbanyak: {tanggal_max_volume.strftime('%Y-%m-%d')}")
