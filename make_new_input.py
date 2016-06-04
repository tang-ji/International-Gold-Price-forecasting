# -*- coding:utf-8 -*-
__author__ = 'Jojo'

import pickle
import numpy as np

name = open('PriceEachDay.pkl')
P = pickle.load(name)
namei = open('Inputs.txt','w')
for i in range(50):
    if len(P[-50+i]) < 50:
        continue
    namei.write(P[-50+i][0] + ' ' + str(np.mean(P[-50+i][1:])) + ' ' + \
                str(np.max(P[-50+i][1:])) + ' ' + str(np.min(P[-50+i][1:])) + '\n')
namei.close()