import numpy as np

# 仮想のDEMデータ (100x100のメッシュで標高を生成)
np.random.seed(42)
dem_data = np.random.uniform(1000, 2500, (100, 100))  # 1000m～2500mの範囲

# DEMデータの簡易的なプロット
import matplotlib.pyplot as plt
plt.imshow(dem_data, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title("Sample DEM Data (North Yatsugatake)")
plt.show()

# 標高に基づく仮想降水データ（高標高は降水量が多い傾向）
precip_data = 0.1 * dem_data + np.random.normal(0, 10, dem_data.shape)

# 降水データのプロット
plt.imshow(precip_data, cmap='Blues')
plt.colorbar(label='Precipitation (mm)')
plt.title("Sample Precipitation Data")
plt.show()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# データ準備 (特徴量: 標高, 目標値: 降水量)
X = dem_data.flatten().reshape(-1, 1)  # 標高データ
y = precip_data.flatten()  # 降水データ

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデル構築
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 予測
y_pred = rf_model.predict(X_test)

# モデル評価
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# 予測データの再構築
y_pred_full = rf_model.predict(X).reshape(dem_data.shape)

# 予測結果の可視化
plt.figure(figsize=(12, 6))

# 実際の降水データ
plt.subplot(1, 2, 1)
plt.imshow(precip_data, cmap='Blues')
plt.colorbar(label='Precipitation (mm)')
plt.title("Actual Precipitation")

# モデルの予測
plt.subplot(1, 2, 2)
plt.imshow(y_pred_full, cmap='Blues')
plt.colorbar(label='Predicted Precipitation (mm)')
plt.title("Predicted Precipitation")

plt.show()

import matplotlib.pyplot as plt

# 予測データの再構築（1Dから2Dに変換）
y_pred_full = rf_model.predict(X).reshape(dem_data.shape)

# 誤差の計算
error_data = precip_data - y_pred_full

# 可視化と保存
plt.figure(figsize=(18, 6))

# 1. 実際の降水データ
plt.subplot(1, 3, 1)
plt.imshow(precip_data, cmap='Blues')
plt.colorbar(label='Precipitation (mm)')
plt.title("Actual Precipitation")

# 2. モデルによる予測
plt.subplot(1, 3, 2)
plt.imshow(y_pred_full, cmap='Blues')
plt.colorbar(label='Predicted Precipitation (mm)')
plt.title("Predicted Precipitation")

# 3. 実際のデータと予測の誤差
plt.subplot(1, 3, 3)
plt.imshow(error_data, cmap='RdYlBu', vmin=-50, vmax=50)
plt.colorbar(label='Error (mm)')
plt.title("Prediction Error (Actual - Predicted)")

plt.tight_layout()

# ファイルに保存
plt.savefig("visualization.png", dpi=300, bbox_inches='tight')
plt.close()

print("可視化結果を 'visualization.png' として保存しました！")