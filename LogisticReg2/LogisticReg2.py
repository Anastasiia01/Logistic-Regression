import sys
from Utils import Utils
import numpy as np

def main():
    utils = Utils()
    data=utils.readData("C:/Users/anast/Documents/Deep Learning/DataSet.csv")
    X=data[:,0:2]
    print(X)
    y=data[:,-1]#reshape to 2d:(100,1) later, now is 1d:(100,)
    X=utils.normalizeData(X)
    X=np.hstack((np.ones((X.shape[0],1)),X))
    utils.plotData(X,y)    
    


if __name__ == "__main__":
    sys.exit(int(main() or 0))