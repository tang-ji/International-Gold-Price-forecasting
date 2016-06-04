# -*- coding:utf-8 -*-
__author__ = 'Jojo'

from Predata import *


def getdata():
    Price = pickle.load(open('PriceEachDay.pkl'))
    return Price

def file2data(filename):
    Price = getdata()
    # Load the data
    fr = open(filename)
    for line in fr.readlines():
        if line[0:6] != 'XAUUSD':
            continue
        # Strip the beginning and the end blank
        lines = line.strip()
        # Separate each line by ','
        lines = lines.split(',')
        date = lines[1][0:8]
        price = float(lines[-1])
        if Price[-1][0] == date:
            Price[-1].append(price)
        if int(Price[-1][0]) < int(date):
            Price.append([date, price])
    name = open('PriceEachDay.pkl', 'w')
    pickle.dump(Price, name)
    name.close()


if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv) == 0:
        kwargs['filename'] = 'XAUUSD.csv'
    if len(sys.argv) > 1:
        kwargs['filename'] = str(sys.argv[1])
    print 'Adding Gold Price data...' + kwargs['filename']
    file2data(**kwargs)
    print 'Renewing the data...'
    Save_HMEachDay()
    Pre_datas()
    print 'Having Finished...'