# -*- coding:utf-8 -*-
__author__ = 'Jojo'

import sys
import os
import re
import math
import string
import datetime
import time
import numpy as np
import pickle

def file2data(filename):
    Date = []
    Price = []
    # Load the data
    fr = open(filename)
    for line in fr.readlines():
        if line[:6] == 'XAUUSD':
            # Strip the beginning and the end blank
            lines = line.strip()
            # Separate each line by ','
            lines = lines.split(',')  
            # Get the values
            Date.append(lines[1])  
            Price.append(float(lines[3]))
    return Date, Price

def MakePriceEachDay(Date, Price):
    Temp = [[]]
    PriceEachDay = []
    for i in range(len(Date)):
        if Temp[0] != Date[i]:
            PriceEachDay.append(Temp)
            Temp = [Date[i]]
        Temp.append(Price[i])
    PriceEachDay.append(Temp)
    name = open('PriceEachDay.pkl','w')
    pickle.dump(PriceEachDay, name)
    name.close()

def Save_HMEachDay():
    Pricefile = open('PriceEachDay.pkl')
    Meanfile = open('MeanEachDay.pkl', 'w')
    LogMeanfile = open('LogMeanEachDay.pkl', 'w')
    TenLogMeanfile = open('TenLogMeanEachDay.pkl', 'w')
    LogMaxfile = open('LogMaxEachDay.pkl', 'w')
    LogMinfile = open('LogMinEachDay.pkl', 'w')
    Price = pickle.load(Pricefile)

    TenLogMean = []
    Mean = []
    LogMean = []
    LogMax = []
    LogMin = []
    for i in range(len(Price)):
        if len(Price[i]) == 1:
            continue
        Mean.append(np.mean(Price[i][1:]))
        LogMean.append(math.log(np.mean(Price[i][1:])))
        LogMax.append(math.log(max(Price[i][1:])))
        LogMin.append(math.log(min(Price[i][1:])))
        if i == 0:
            TenLogMean.append(math.log(Mean[0]))
        else:
            TenLogMean.append(math.log(np.mean(Mean[-11:-1])))

    pickle.dump(Mean, Meanfile)
    pickle.dump(TenLogMean, TenLogMeanfile)
    pickle.dump(LogMax, LogMaxfile)
    pickle.dump(LogMin, LogMinfile)
    pickle.dump(LogMean, LogMeanfile)
    Meanfile.close()
    LogMaxfile.close()
    LogMinfile.close()
    LogMeanfile.close()
    TenLogMeanfile.close()

def Pre_datas():
    Pricefile = open('PriceEachDay.pkl')
    Meanfile = open('MeanEachDay.pkl')
    LogMeanfile = open('LogMeanEachDay.pkl')
    TenLogMeanfile = open('TenLogMeanEachDay.pkl')
    LogMaxfile = open('LogMaxEachDay.pkl')
    LogMinfile = open('LogMinEachDay.pkl')
    Price = pickle.load(Pricefile)
    Mean = pickle.load(Meanfile)
    LogMean = pickle.load(LogMeanfile)
    TenLogMean = pickle.load(TenLogMeanfile)
    LogMax = pickle.load(LogMaxfile)
    LogMin = pickle.load(LogMinfile)

    Inputs = []
    for i in range(len(Price))[30:-20]:
        L1 = [Mean[i]/2000*0.03] + [math.log(np.mean(Mean[i-9:i-4])/np.mean(Mean[i-19:i-9]))] + list(np.array(LogMean[i-4:i+1])-np.array(TenLogMean[i-4:i+1]))
        L2 = [max(Price[i][1:])/2000*0.03] + [max(LogMax[i-9:i-4])-TenLogMean[i-9]] + list(np.array(LogMax[i-4:i+1])-np.array(TenLogMean[i-4:i+1]))
        L3 = [min(Price[i][1:])/2000*0.03] + [min(LogMin[i-9:i-4])-TenLogMean[i-9]] + list(np.array(LogMin[i-4:i+1])-np.array(TenLogMean[i-4:i+1]))
        InputsTemp = [[L1, L2, L3]]
        Inputs.append(InputsTemp)

    Outputs = []
    PO = []
    for i in range(len(Price))[30:-20]:
        #L1 = list(np.array(LogMean[i+1:i+4])-np.array(LogMean[i:i+3])) + [math.log(np.mean(Mean[i+4:i+9])/Mean[i+3])] + [math.log(np.mean(Mean[i+9:i+14])/Mean[i+8])]
        #L2 = Max[i+1:i+6] + [max(Max[i+6:i+11])] + [max(Max[i+11:i+16])]
        #L3 = Min[i+1:i+6] + [min(Min[i+6:i+11])] + [min(Min[i+11:i+16])]
        L1 = list(np.array(LogMean[i+1: i+6])-np.array(TenLogMean[i+1: i+6]))
        OutputsTemp = L1
        Outputs.append(OutputsTemp)

    Outputs2 = []
    for i in range(len(Price))[30:-20]:
        if Mean[i+1] >= Mean[i]:
            L1 = 1
        if Mean[i+1] < Mean[i]:
            L1 = 0
        OutputsTemp = L1
        Outputs2.append(OutputsTemp)


    Inputs = np.array(Inputs,dtype=np.float32)
    Outputs = np.array(Outputs,dtype=np.float32)
    Outputs2 = np.array(Outputs2,dtype=np.uint8)


    def Change_IO(Inputs, Outputs):
        for i in range(len(Inputs)):
            for j in [0,1,2]:
                Inputs[i][0][j] = (Inputs[i][0][j]+0.03)/0.06
                for k in range(len(Inputs[i][0][j])):
                    if Inputs[i][0][j][k]>1:
                        Inputs[i][0][j][k]=1
                    if Inputs[i][0][j][k]<0:
                        Inputs[i][0][j][k]=0
    
        for i in range(len(Outputs)):
            Outputs[i] = (Outputs[i]+0.03)/0.06
            for k in range(len(Outputs[i])):
                if Outputs[i][k]>1:
                    Outputs[i][k]=1
                if Outputs[i][k]<0:
                    Outputs[i][k]=0
        return Inputs,Outputs

    Inputs, Outputs = Change_IO(Inputs, Outputs)
    namei = open('Inputs.pkl','w')
    nameo = open('Outputs.pkl','w')
    nameo2 = open('Outputs2.pkl','w')
    pickle.dump(Outputs2, nameo2)
    pickle.dump(Inputs, namei)
    pickle.dump(Outputs, nameo)
    namei.close()
    nameo.close()
    nameo2.close()

def MakeMeanEachTenDays():
    Meanfile = open('MeanEachDay.pkl')
    Mean = pickle.load(Meanfile)
    Temp = []
    TenMean = []
    for i in range(len(Mean)):
        Temp.append(Mean[i])
        TenMean.append(np.mean(Temp[-10:]))
    name = open('MeanEachTenDays.pkl','w')
    pickle.dump(TenMean, name)
    name.close()

if __name__ == '__main__':
    print 'Loading Gold Price data...\n'
    D, P = file2data('XAUUSD.txt')
    print 'Classfying Gold Price for each day...\n'
    MakePriceEachDay(D, P)
    print 'Preparing the data...\n'
    Save_HMEachDay()
    Pre_datas()
    print 'Having Finished...\n'