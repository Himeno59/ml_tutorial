import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from softmax_regression import SoftmaxRegression

# データのインポート
iris = load_iris()
X = iris.data # (150,4)
t = iris.target # (150,)

# 特徴量の標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# データ分割
X_train, X_test, t_train, t_test = train_test_split(X_scaled, t, test_size=0.2, random_state=42, stratify=t)

# モデル設定
model = SoftmaxRegression(lr=0.1)
model.fit(X_train, t_train)

# 学習
model.train(num_iter=5000)

# 評価
t_pred = model.predict(X_test)
accuracy = np.mean(t_pred == t_test)
print(f"Test Accuracy: {accuracy:.4f}")
