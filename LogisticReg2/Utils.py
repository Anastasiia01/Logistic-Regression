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
        np.random.seed(12)
        num=50
        class0=np.random.multivariate_normal([4,3.5],[[1,0.75],[0.75, 1]],size=50)
        class0=np.hstack((class0,np.zeros((class0.shape[0],1))))
        class1=np.random.multivariate_normal([4,1.5],[[1,0.75],[0.75, 1]],size=50)
        class1=np.hstack((class1,np.ones((class1.shape[0],1))))
        data=np.vstack((class0,class1))
        return(data)



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
        # do a scatter plot of the data
        plt.scatter(x0[:,1],x0[:,2],c='b',marker='o',label="y=0")
        plt.scatter(x1[:,1],x1[:,2],c='r',marker='X',label="y=1")

        #plot the fitted line
        x1 = np.arange(0, 1.1, 0.1)
        #hyp=beta0+beta1*x1+beta2*x2=X*beta
        x2=-(beta[0]+beta[1]*x1)/beta[2]
        plt.plot(x1,x2,linewidth=0.8,c='g',label='reg.line')

        plt.title('Logistic Regression to classify binary data')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.show()
