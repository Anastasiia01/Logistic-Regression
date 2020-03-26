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
        return None

    def plotData(self,X,y):
        y=y.reshape(100,)
        x0=X[y==0]
        x1=X[y==1]
        plt.scatter(x0[:,1],x0[:,2],c='b',marker='o',label="y=0")
        plt.scatter(x1[:,1],x0[:,2],c='r',marker='X',label="y=1")
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.show()

    def plotResult(self, X,y,beta):
        return None
