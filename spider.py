# -*- coding:utf-8 -*-
__author__ = 'Jojo'

import re
import requests
import string
import time
import sys
import pickle

def delch(li, ch):
    temp = []
    for i in range(len(li)):
        temp += li[i].split(ch)
    temp2 = []
    for j in range(len(temp)):
        if len(temp[j]) > 0:
            temp2 += [temp[j]]
    return temp2

def filter(inf_in):
    inf = delch([inf_in], '</p><p>')
    inf = delch(inf, '<strong>')
    inf = delch(inf, '<div class="snsbox-inner">')
    inf = delch(inf, '<p>')
    inf = delch(inf, '<div class="sidesnstip">')
    inf = delch(inf, '</span>')
    inf = delch(inf, '<div class="snsbox">')
    inf = delch(inf, '<a target="_blank" href="')
    inf = delch(inf, '">')
    inf = delch(inf, '<div class="snszone" data-url="')
    inf = delch(inf, '<span>')
    inf = delch(inf, '</strong>')
    inf = delch(inf, u'\u63d0\u793a')
    inf = delch(inf, u'\uff1a')
    inf = delch(inf, u'\u8bf7\u70b9\u51fb\u4e0a\u65b9\u84dd\u8272\u5c0f\u5b57\u5173\u6ce8\u6211\u4eec\u5662\uff0c\u6da8\u59ff\u52bf\u540e\u80fd\u8d5a\u5927\u94b1\uff01')
    inf = delch(inf, '<em>')
    inf = delch(inf, '</em>')
    inf = delch(inf, '<b>')
    inf = delch(inf, '</b>')
    inf = delch(inf, '<u>')
    inf = delch(inf, '</u>')
    return inf

class inf_list:
    class day_inf:
        def __init__(self, time, value, sum_up, sum_down):
            self.time = time
            self.value = value
            self.sum_up = sum_up
            self.sum_down = sum_down
    def __init__(self):
        self.inf = []
    def add(self, time, value, sum_up, sum_down):
        if len(self.inf) == 0:
            self.inf.append(self.day_inf(time, value, sum_up, sum_down))
        else:
            rep = 0
            for i in range(len(self.inf)):
                if time == self.inf[i].time:
                    self.inf[i].value += value
                    self.inf[i].sum_up += sum_up
                    self.inf[i].sum_down += sum_down
                    rep = 1
            if rep == 0:
                self.inf.append(self.day_inf(time, value, sum_up, sum_down))
            self.inf.sort(lambda x, y: cmp(x.time, y.time), reverse=True)
    def show(self):
        for i in range(len(self.inf)):
            print self.inf[i].time + ' ' + str(self.inf[i].value) + ' ' \
                  + str(self.inf[i].sum_up) + ' ' + str(self.inf[i].sum_down)
    def save(self, name):
        file = open(name, 'w')
        for i in range(len(self.inf)):
            file.write(self.inf[i].time + ' ' + str(self.inf[i].value) + ' ' \
                  + str(self.inf[i].sum_up) + ' ' + str(self.inf[i].sum_down) + '\n')
        file.close()
    def load(self, name):
        file = open(name)
        fr = file.readlines()
        for i in range(len(fr)):
            f = fr[i].strip().split(' ')
            time = f[0]
            value = [j.strip('[]').strip(',').strip("'") for j in f[1:-2]]
            try:
                value.remove('')
            except ValueError or AttributeError:
                pass
            sum_up = int(f[-2])
            sum_down = int(f[-1])
            self.add(time, value, sum_up, sum_down)
    def set_max(self, max_value):
        for j in range(len(self.inf)):
            error_label = []
            for i in range(len(self.inf[j].value)):
                if float(self.inf[j].value[i]) >= float(max_value):
                    error_label.append(self.inf[j].value[i])
            if len(error_label) > 0:
                for i in error_label:
                    self.inf[j].value.remove(i)




def get_toutiao(n = 300):
    print 'Connecting to toutiao.com...'
    news_numbers = str(n)
    links1 = requests.get('http://toutiao.com/search_content/?offset=0&format=json&keyword=%E5%9B%BD%E9%99%85+%E9%BB%84%E9%87%91&autoload=true&count=' + news_numbers + '&_=1460862899094')
    links2 = requests.get('http://toutiao.com/search_content/?offset=0&format=json&keyword=%E5%9B%BD%E9%99%85+%E9%BB%84%E9%87%91+%E9%A2%84%E6%B5%8B&autoload=true&count=' + news_numbers + '&_=1460864327503')
    url = re.findall('item_source_url": "(.*?)",', links1.text, re.S)
    news_links = []

    print 'Getting the urls...'
    for each_new in url:
        news_links.append('http://toutiao.com' + str(each_new))

    url = re.findall('item_source_url": "(.*?)",', links2.text, re.S)
    for each_new in url:
        news_links.append('http://toutiao.com' + str(each_new))

    print 'Downloading the news...'
    news = []
    total = float(len(news_links))
    try:
        for index in range(len(news_links)):
            news.append(requests.get(news_links[index]).text)
            print("  Finished:\t{:.1f}%".format(float(index)*100 / total))
    finally:
        print 'Searching the key words...'
        inf = inf_list()
        for index in range(len(news)):
            inf_time = re.findall('<span class="time">(.*?)</span>', news[index])
            try:
                timetemp = str(inf_time[0])[0:10]
            except IndexError:
                continue
            keywords = []
            sum_up = 0
            sum_down = 0
            # key words：金价...会跌至...
            keywords += re.findall(u'\u91d1\u4ef7.*?\u4f1a\u8dcc\u81f3(\d\d\d\d)', news[index])
            # key words：黄金...重点关注...
            keywords += re.findall(u'\u9ec4\u91d1.*?\u91cd\u70b9\u5173\u6ce8(\d\d\d\d)', news[index])
            # key words：黄金...看空到...
            keywords += re.findall(u'\u9ec4\u91d1.*?\u770b\u7a7a\u5230(\d\d\d\d)', news[index])
            # key words：黄金...有可能突破...
            keywords += re.findall(u'\u9ec4\u91d1.*?\u6709\u53ef\u80fd\u7a81\u7834(\d\d\d\d)', news[index])
            # key words: 黄金...有望涨至...
            keywords += re.findall(u'\u9ec4\u91d1.*?\u6709\u671b\u6da8\u81f3(\d\d\d\d)', news[index])
            # key words: 黄金...预测价...
            keywords += re.findall(u'\u9ec4\u91d1.*?\u9884\u6d4b\u4ef7(\d\d\d\d)', news[index])
            # key words: 黄金...目标位是...
            keywords += re.findall(u'\u9ec4\u91d1.*?\u76ee\u6807\u4f4d\u662f(\d\d\d\d)', news[index])
            # key words: 黄金...目标...
            keywords += re.findall(u'\u9ec4\u91d1.*?\u76ee\u6807(\d\d\d\d)', news[index])
            # key words: 黄金...将继续走高
            sum_up += len(re.findall(u'\u9ec4\u91d1.*?\u5c06\u7ee7\u7eed\u8d70\u9ad8', news[index]))
            # key words: 黄金...我们仍看多
            sum_up += len(re.findall(u'\u9ec4\u91d1.*?\u6211\u4eec\u4ecd\u770b\u591a', news[index]))
            # key words: 看好...黄金...涨势
            sum_up += len(re.findall(u'\u770b\u597d.*?\u9ec4\u91d1.*?\u6da8\u52bf', news[index]))
            # key words: 持有黄金是明智的
            sum_up += len(re.findall(u'\u6301\u6709\u9ec4\u91d1\u662f\u660e\u667a\u7684', news[index]))
            # key words: 黄金...继续走强
            sum_up += len(re.findall(u'\u9ec4\u91d1.*?\u7ee7\u7eed\u8d70\u5f3a', news[index]))
            # key words: 支撑美元
            sum_down += len(re.findall(u'\u652f\u6491\u7f8e\u5143', news[index]))
            # key words: 支撑黄金
            sum_up += len(re.findall(u'\u652f\u6491\u9ec4\u91d1', news[index]))
            # key words: 黄金在近期内将走弱
            sum_down += len(re.findall(u'\u9ec4\u91d1\u5728\u8fd1\u671f\u5185\u5c06\u8d70\u5f31', news[index]))
            # key words: 黄金在近期内将走强
            sum_up += len(re.findall(u'\u9ec4\u91d1\u5728\u8fd1\u671f\u5185\u5c06\u8d70\u5f3a', news[index]))
            # key words: 美元将下跌
            sum_up += len(re.findall(u'\u7f8e\u5143\u5c06\u4e0b\u8dcc', news[index]))
            # key words: 黄金将下跌
            sum_down += len(re.findall(u'\u9ec4\u91d1\u5c06\u4e0b\u8dcc', news[index]))
            # key words: 美元将上涨
            sum_down += len(re.findall(u'\u7f8e\u5143\u5c06\u4e0a\u6da8', news[index]))
            # key words: 黄金将上涨
            sum_up += len(re.findall(u'\u9ec4\u91d1\u5c06\u4e0a\u6da8', news[index]))
            # key words: 推动黄金价格上涨
            sum_up += len(re.findall(u'\u63a8\u52a8\u9ec4\u91d1\u4ef7\u683c\u4e0a\u6da8', news[index]))
            # key words: 增...黄金仓位
            sum_up += len(re.findall(u'\u589e.*?\u9ec4\u91d1\u4ed3\u4f4d', news[index]))
            # key words: 减...黄金仓位
            sum_down += len(re.findall(u'\u51cf.*?\u9ec4\u91d1\u4ed3\u4f4d', news[index]))
            # key words: 美元走强
            sum_down += len(re.findall(u'\u7f8e\u5143\u8d70\u5f3a', news[index]))
            # key words: 美元走弱
            sum_up += len(re.findall(u'\u7f8e\u5143\u8d70\u5f31', news[index]))
            # key words: 建议做多黄金
            sum_up += len(re.findall(u'\u5efa\u8bae\u505a\u591a\u9ec4\u91d1', news[index]))
            # key words: 建议做空黄金
            sum_down += len(re.findall(u'\u5efa\u8bae\u505a\u7a7a\u9ec4\u91d1', news[index]))
            # key words: 黄金...保持看多
            sum_up += len(re.findall(u'\u9ec4\u91d1.*?\u4fdd\u6301\u770b\u591a', news[index]))
            # key words: 黄金的季节性放缓
            sum_down += len(re.findall(u'\u9ec4\u91d1.*?\u653e\u7f13', news[index]))
            # key words: 金价...走弱
            sum_down += len(re.findall(u'\u91d1\u4ef7.*?\u8d70\u5f31', news[index]))
            # key words: 金价...走强
            sum_up += len(re.findall(u'\u91d1\u4ef7.*?\u8d70\u5f3a', news[index]))
            # key words: 金价...下跌
            sum_down += len(re.findall(u'\u91d1\u4ef7.*?\u4e0b\u8dcc', news[index]))
            # key words: 金价...上涨
            sum_up += len(re.findall(u'\u91d1\u4ef7.*?\u4e0a\u6da8', news[index]))
            # key words: 黄金...市场...承压
            sum_down += len(re.findall(u'\u9ec4\u91d1.*?\u5e02\u573a.*?\u627f\u538b', news[index]))
            if keywords == [] and sum_up == 0 and sum_down == 0:
                continue
            inf.add(timetemp, [k.encode('utf8') for k in keywords], sum_up, sum_down)

        print 'Saving...'
        date = time.ctime(time.time()).split(' ')
        inf.save('data/' + date[2] + '.txt')
        label = date[-1] + date[1] + date[2]
        namen = open('data/' + label + '.pkl', 'w')
        pickle.dump(news, namen)
        namen.close()

        news_forecast = []
        f = open('data/' + label + '.txt', 'w')
        for index in range(len(news)):
            inf_time = re.findall('<span class="time">(.*?)</span>', news[index], re.S)
            if len(inf_time) == 0:
                continue
            else:
                timetemp = str(inf_time[0])[0:10]
            # key words: ...黄金...
            a = re.findall(u'>(.*?\u9ec4\u91d1.*?)<', news[index])
            # key words: ...美元...
            b = re.findall(u'>(.*?\u7f8e\u5143.*?)<', news[index])
            if len(a) == 0 and len(b) == 0:
                continue
            f.write('\n' + timetemp + '\n')
            inf_temp = []
            if len(a) > 0:
                for i in range(len(a)):
                    aa = filter(a[i])
                    for j in range(len(aa)):
                        f.write(aa[j].encode("utf-8") + '\n')
                        inf_temp.append(aa[j])
            if len(b) > 0:
                for i in range(len(b)):
                    bb = filter(b[i])
                    for j in range(len(bb)):
                        f.write(bb[j].encode("utf-8") + '\n')
                        inf_temp.append(bb[j])
            news_forecast.append(inf_temp)

        f.close()
        print("Having saved:\t{:.1f}%".format(float(index)*100 / total))



def get_forecast():
    links4 = requests.get('http://www.forecasts.org/gold.htm')
    Inf = re.findall(u'to be (.*) per troy ounce for (.* 2016)', links4.text)

if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['n'] = int(sys.argv[1])
    get_toutiao(**kwargs)
    print "Successfully finished..."