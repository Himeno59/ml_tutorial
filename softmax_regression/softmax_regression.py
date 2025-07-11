import numpy as np

class SoftmaxRegression:
    def __init__(self, W=None, b=None, lr=0.1, verbose=False):
        # 自分で設定
        self.W = W # 次元数×クラス数
        self.b = b # データ数
        self.lr = lr
        self.verbose = verbose

        # データ
        self.X = None # データ数×次元
        self.t = None # データ数×1
        self.t_one_hot = None

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) # e^zが大きくならないように調整
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot(self, t, num_classes):
        # one-of-K符号化法
        # tは一次元配列想定
        one_hot = np.zeros((t.shape[0],num_classes)) # データ数×クラス数
        one_hot[np.arange(t.shape[0]),t] = 1 # 各行ごとのt(クラスインデックス)番目に1を入れていく
        return one_hot

    def fit(self, X, t):
        self.X = X
        self.t = t
        n, d = self.X.shape
        num_classes = np.max(t) + 1
        self.t_one_hot = self.one_hot(t, num_classes)
        
        if self.W is None:
            self.W = np.zeros((d, num_classes))
        else:
            if self.W.shape != (d, num_classes):
                raise ValueError(f"Wの形状が不正です。期待される形状: {(d, num_classes)}、渡された形状: {self.W.shape}")

        if self.b is None:
            self.b = np.zeros(num_classes)
        else:
            if self.b.shape != (num_classes,):
                raise ValueError(f"bの形状が不正です。期待される形状: {(num_classes,)}、渡された形状: {self.b.shape}")

        if self.verbose:
            print(f"X.shape: {self.X.shape}")
            print(f"W.shape: {self.W.shape}")
            print(f"b.shape: {self.b.shape}")
            print(f"t.shape: {self.t_one_hot.shape}")
            
    def train(self, num_iter=1000):
        for i in range (num_iter):
            P = self.softmax(np.dot(self.X, self.W)+self.b) # ブロードキャストが適用される
            grad_W = np.dot(self.X.T, (P-self.t_one_hot))
            grad_b = np.mean(P-self.t_one_hot, axis=0)
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

            if self.verbose and i % 100 == 0:
                loss = -np.mean(np.sum(self.t_one_hot * np.log(P+1e-15), axis=1))
                print(f"Iteration {i}: loss={loss:.4f}")

    def predict(self, X):
        return np.argmax(self.softmax(np.dot(X, self.W)+self.b), axis=1)
