from simple_perceptron import * # 同じ階層にある前提

def main():
    x = np.array([1,2,3])
    w = np.array([1,2,3])
    b = 1
    perceptron = SimplePerceptron(x,w,b)

    perceptron.calc()
    
if __name__ == "__main__":
    main()
