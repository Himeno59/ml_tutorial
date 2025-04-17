import numpy as np
import matplotlib.pyplot as plt
from simple_perceptron import * # 同じ階層にある前提
from sklearn.datasets import load_iris

def main():
    ### データ ###
    # irisデータセット
    iris = load_iris()
    X = iris.data[:100, :2]
    T = iris.target[:100]

    ### インスタンス生成 ###
    w = np.zeros(2)
    b = 0
    perceptron = perceptron = SimplePerceptron(len(X[0]),w,b) 

    ### 学習 ###
    for epoch in range(10):
        for x,t in zip(X,T): 
            perceptron.train(x,t)

    ## visualize ##
    for i in range(len(X)):
        if T[i] == 1:
            plt.scatter(X[i][0], X[i][1], color='blue', label='Class 1' if i==0 else "")
        else:
            plt.scatter(X[i][0], X[i][1], color='red', label='Class 0' if i==2 else "")
            
    # 分離直線を描画
    x1_vals = np.linspace(-3, 3, 100)
    if w[1] != 0:
        x2_vals = -(w[0] * x1_vals + b) / w[1]
        plt.plot(x1_vals, x2_vals, color='green', label='Decision Boundary')
    else:
        plt.axvline(-b/w[0], color='green', label='Decision Boundary')
        
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.title("Perceptron Classification Result")
    plt.show()
    
if __name__ == "__main__":
    main()
