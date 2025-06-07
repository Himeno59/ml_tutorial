import numpy as np
import matplotlib.pyplot as plt
from simple_perceptron import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

## data ##
iris = load_iris()
print(f"data: ({iris.data.shape[0]}, {iris.data.shape[1]})")

## 線形分離のために最初の2種類を取り出す ##
X = iris.data[:100,:2]
T = iris.target[:100]
X_train, X_test, T_train, T_test = train_test_split(X,T,test_size=0.3,random_state=42, stratify=T)
print(f"X: ({X.shape[0]}, {X.shape[1]})")
print(f"T: ({len(T)})")

## param ##
w = np.zeros(X.data.shape[1])
b = 0
lr = 0.1
print(f"w: {w}, b: {b}, lr: {lr}")
    
## create instance ##
perceptron = SimplePerceptron(X.shape[1],w,b,lr)

## learning ##
for epoch in range(10000):
    for x,t in zip(X_train,T_train): 
        perceptron.train(x,t)
        
## test ##
accuracy = perceptron.test(X_test, T_test)
print("accuracy: ",accuracy)
            
## visualize ##
## 特徴量が2次元で直線で書ける場合のみ
for i in range(len(X)):
    if T[i] == 1:
        plt.scatter(X[i][0], X[i][1], color='blue', label='Class 1' if i==0 else "")
    else:
        plt.scatter(X[i][0], X[i][1], color='red', label='Class 0' if i==2 else "")

plt.xlim(auto=True)
xlim = plt.xlim()
x1_vals = np.linspace(xlim[0], xlim[1], 100)
if perceptron.w[1] != 0:
    x2_vals = -(perceptron.w[0] * x1_vals + perceptron.b) / perceptron.w[1]
    plt.plot(x1_vals, x2_vals, color='green', label='Decision Boundary')
else:
    plt.axvline(-perceptron.b/perceptron.w[0], color='green', label='Decision Boundary')

print(f"w: {w}")

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.title("Perceptron Classification Result")
plt.show()    
