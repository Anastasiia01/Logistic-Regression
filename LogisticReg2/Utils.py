import csv
import numpy as np
import matplotlib.pyplot as plt

class Utils():
    def readData(self, file_path):
        with open(file_path,'r')as out:
            reader=csv.reader(out) #delimiter=',' is default
            #headers = next(reader)if first row is headers
            data = np.array(list(reader)).astype(float)
        return data

    def readDataRandom(self):
        return None

    def normalizeData(self,X):
        max=np.max(X,axis=0)
        min=np.min(X,axis=0)
        norm_X=1-((max-X)/(max-min))
        return norm_X


    def plotData(self,X,y):
        Y=y.reshape(100,)
        x0=X[Y==0]
        x1=X[Y==1]
        plt.scatter(x0[:,1],x0[:,2],c='b',marker='o',label="y=0")
        plt.scatter(x1[:,1],x1[:,2],c='r',marker='X',label="y=1")
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.show()

    def plotResult(self, X,y,beta):
        Y=y.reshape(100,)
        x0=X[Y==0]
        x1=X[Y==1]
        plt.scatter(x0[:,1],x0[:,2],c='b',marker='o',label="y=0")
        plt.scatter(x1[:,1],x1[:,2],c='r',marker='X',label="y=1")
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.show()
