import sys
from Utils import Utils
import numpy as np

def sigmoid(t):
    return 1/(1+np.exp(-t))

def lossFunction(X,y,beta):
    h=sigmoid(X@beta)
    L=-(y*np.log(h)+(1-y)*np.log(1-h))
    return np.sum(L)

def gradBeta(X,y,beta):
    h=sigmoid(X@beta)
    grad=X.T@(h-y)
    return grad

def trainLogRegression(X,y,beta,numIter,alpha=0.01):
    loss=lossFunction(X,y,beta)
    for i in range(numIter):
        beta=beta-alpha*gradBeta(X,y,beta)
        if(i%10==0):
            loss=lossFunction(X,y,beta)
            print("i= ",i,"loss= ",loss)
    return beta


def main():
    utils = Utils()
    data=utils.readData("C:/Users/anast/Documents/Deep Learning/DataSet.csv")
    #data=utils.readDataRandom()
    utils.readDataRandom()
    X=data[:,0:2]
    y=data[:,-1].reshape(100,1)#initially y is (100,) matrix
    X=utils.normalizeData(X)
    X=np.hstack((np.ones((X.shape[0],1)),X))
    beta=np.zeros((X.shape[1],1))#is 3x1 matrix
    num=1000
    beta=trainLogRegression(X,y,beta,num)   
    print(beta)
    utils.plotResult(X,y,beta)      


if __name__ == "__main__":
    sys.exit(int(main() or 0))