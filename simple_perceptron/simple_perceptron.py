import numpy as np

def sgn(x):
    return 1 if x >= 0 else 0

class SimplePerceptron:
    def __init__(self, input_size, w=None, b=0, lr=0.1):
        
        if not isinstance(w, np.ndarray):
            raise TypeError("wはNumPy配列である必要があります。")
    
        self.w = w if w is not None else np.zeros(input_size)
        self.b = b
        self.lr = lr
        self.y = None
        
    def calc(self, x):
        return sgn(np.dot(self.w,x)+self.b)

    def train(self, x, t): #誤り訂正法
        self.y = self.calc(x)
        self.w += self.lr * (t - self.y)*x
        self.b += self.lr * (t - self.y)

    def test(self, X, T):
        correct = 0;
        for x, t in zip(X, T):
            if self.calc(x) == t:
                correct += 1
        accuracy = correct / len(T)
        return accuracy
