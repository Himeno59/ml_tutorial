import numpy as np

def sgn(x):
    return 1 if x >= 0 else 0

class SimplePerceptron:
    def __init__(self, input_size, w=None, b=0):
        
        if not isinstance(w, np.ndarray):
            raise TypeError("wはNumPy配列である必要があります。")
    
        self.w = w if w is not None else np.zeros(input_size)
        self.b = b
        self.y = None
        
    def calc(self, x):
        return sgn(np.dot(self.w,x)+self.b)

    def train(self, x, t):
        self.y = self.calc(x)
        self.w += (t - self.y)*x
        self.b += (t - self.y)
