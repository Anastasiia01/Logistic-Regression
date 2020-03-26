import sys
from Utils import Utils
import numpy as np

def main():
    utils = Utils()
    data=utils.readData("C:/Users/anast/Documents/Deep Learning/DataSet.csv")
    X=np.hstack((np.ones((data.shape[0],1)),data[:,0:2]))
    y=data[:,-1].reshape(100,1)
    utils.plotData(X,y)    


if __name__ == "__main__":
    sys.exit(int(main() or 0))