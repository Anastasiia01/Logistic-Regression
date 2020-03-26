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
    def plotResult(self, X,y,beta):
        return None
