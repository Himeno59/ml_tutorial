import numpy as np

def sgn(x):
    return 1 if x >= 0 else 0

class SimplePerceptron:
    def __init__(self, x, w, b):
        # check value
        if not (isinstance(x, np.ndarray) and isinstance(w, np.ndarray)):
            raise TypeError("xとwはNumPy配列である必要があります。")
        if x.ndim != 1 or w.ndim != 1:
            raise ValueError("xとwは1次元のベクトルでなければなりません。")
        if len(x) != len(w):
            raise ValueError("xとwは同じ長さでなければなりません。")
        self.x = x
        self.w = w
        self.b = b

        self.output = None
        

    def calc(self):
        self.output = np.dot(self.w,self.x)+self.b
        print(self.output)
        return self.output

    def predict(self):
        print(sgn(self.output))
        return sgn(self.output)
