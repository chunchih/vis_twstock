import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter, WeekdayLocator, \
    DayLocator, MONDAY, date2num, num2date
from matplotlib.finance import candlestick_ohlc
import datetime
import numpy as np
from numpy import genfromtxt
import os
import pandas as pd
import time

def get_stock(f_name):
    stock_list = np.array([str(int(s)) for s in np.loadtxt(f_name)])
     
    stock_file = dict()
    for i,j,k in os.walk('data'):
        for kk in k:
            if kk.lstrip('0').rstrip('.csv') in stock_list:
                stock_file[kk.lstrip('0').rstrip('.csv')] = os.path.join(i,kk)
    return stock_list, stock_file

def processDate(ch_time):
    t = ch_time.split("/")
    return "%d%s%s"%(int(t[0])+1911,t[1].zfill(2),t[2].zfill(2))

def getData_by_stock(stock_file, stock_no, end_dt):
    data = pd.read_csv(stock_file[str(stock_no)],sep='\n')
    date_r = []
    volume = []
    price = []    
    for e, d in enumerate(data.values):
        try:
            detail = d[0].split(",")
            if "--" in detail or "---" in detail:
                continue
            if processDate(detail[0]) == "20180113":
                continue
            # ohlc
            date_r += [processDate(detail[0])]
            volume += [int(float(detail[1]))]
            price += [[float(detail[3]), float(detail[4]), float(detail[5]), float(detail[6])]]
        except:
            print "error", detail
            while True:
                a = 1
        
    date_r = np.array(date_r); volume = np.array(volume); price = np.array(price)
    
    date_correct = True
    idx = -1
    while end_dt >= 19800101:
        if len(np.where(date_r==str(end_dt))[0]) == 0: # not exists
            end_dt-=1
            date_correct = False
        else:
            idx = np.where(date_r==str(end_dt))[0][0]
            break

    if idx == -1:
        return [], [], [], False
    date_r = date_r[:idx+1]; volume = volume[:idx+1]; price = price[:idx+1]
    d2n = date2num(datetime.date(end_dt//10000,(end_dt%10000)//100,end_dt%100))
    date_p = np.array([[float(i)] for i in range(int(d2n)-len(date_r)+1, int(d2n)+1, 1)])
    return date_p, volume, price, date_correct
    
def MA(close, period, duration):
    ma = []
    for i in range(duration-1,-1,-1):
        ma += [round(np.mean([float(d) for d in close[len(close)-i-period:len(close)-i]]),2)]
    return ma

def draw_volume(volume, date, color, duration, ax):
    data = np.array([float(d) for d in volume[-duration:]])  
    ax.bar(date[-duration:], data, color=color[-duration:])
    
def draw_plot(date, volume, price, duration):
    
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    dayFormatter = DateFormatter('%d')      # e.g., 12
    weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12

    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot2grid((6,4),(1,0),rowspan=4, colspan=4)
    
    K_graph = np.concatenate((date, price), axis=1)[-duration:]   
    
    fig.set_size_inches(20,10)
    candlestick_ohlc(ax, K_graph, width=0.6, colorup='r', colordown='black')
    
    close_p = price[:,3]
    open_p = price[:,0]
    ma10 = MA(close_p, 10, duration)
    ax.plot(date[-duration:], ma10, c='green', lw=0.8)
    ma20 = MA(close_p, 20, duration)
    ax.plot(date[-duration:], ma20, c='brown', lw=0.8)
    ma60 = MA(close_p, 60, duration)
    ax.plot(date[-duration:], ma60, c='purple', lw=0.8)
    plt.show()
    
    fig = plt.figure(figsize=(20,3))
    ax1 = plt.subplot2grid((6,4),(1,0),rowspan=4, colspan=4)
    m = close_p >= open_p
    v_color = []
    for mm in m:
        if mm == True:
            v_color.append('r')
        else:
            v_color.append('black')
            
    draw_volume(volume, date, v_color, duration, ax1)
    plt.show()

def predict_item(item, predict, current):
    print item, predict, predict-current

def predict_price(open_p, close_p, high_p, previous_day=0, display=False):
    if previous_day != 0:
        open_p = open_p[:previous_day]; close_p = close_p[:previous_day]; high_p = high_p[:previous_day]
    item = [">10MA", ">20MA", ">60MA", "RSI GX", "10-20 GX", "20-60 GX", "10-60 GX", "H.Y. Highest"] #"diverge", "Previous High Peak"]
    from collections import OrderedDict
    fullfill = []
    print_info = OrderedDict()
    print_info['done'] = 0
    print_info['<2%'] = 0
    print_info['2%~5%'] = 0
    print_info['5%~8%'] = 0
    print_info['>8%'] = 0
    info = dict()

    for i in item:
        
        if ">10MA" == i:
            bottleneck = MA(close_p, 9, 1)[-1]
        elif ">20MA" == i:    
            bottleneck = MA(close_p, 19, 1)[-1]
        elif ">60MA" == i:    
            bottleneck = MA(close_p, 59, 1)[-1]
        elif "diverge" == i:
            bottleneck = max(2*close_p[-10]-close_p[-20], 1.2*close_p[-10]-0.2*close_p[-60], 1.5*close_p[-20]-0.5*close_p[-60])
        elif "RSI GX" == i:
            up = (close_p >= open_p)*close_p
            down = (close_p < open_p)*close_p
            RSI6 = np.sum(up[-6:])/(np.sum(up[-6:])+np.sum(down[-6:]))
            RSI12 = np.sum(up[-12:])/(np.sum(up[-12:])+np.sum(down[-12:]))
            if RSI6 >= RSI12:
                bottleneck = close_p[-1]
            else:
                if np.sum(up[-11:-5])+np.sum(down[-11:-5]) < 0:
                    continue
                bottleneck = (np.sum(up[-11:])*np.sum(down[-5:])-np.sum(up[-5:])*np.sum(down[-11:]))/(np.sum(up[-11:-5])+np.sum(down[-11:-5]))
        elif "10-20 GX" == i:
            bottleneck = np.sum(close_p[-19:-9])-np.sum(close_p[-9:])
            
        elif "20-60 GX" == i:
            bottleneck = np.sum(close_p[-59:-19])/2.0-np.sum(close_p[-19:])

        elif "10-60 GX" == i:
            bottleneck = np.sum(close_p[-59:-9])/5.0-np.sum(close_p[-9:])

        elif "H.Y. Highest" == i:
            bottleneck = max(close_p[-280:])
        elif "Previous High Peak" == i:
            it = -2
            bottleneck = 0
            while it > -250:
                if high_p[it] > high_p[it-1] and high_p[it] > high_p[it+1]:
                    bottleneck = high_p[it]
                    if high_p[it] >= close_p[-1]:
                        bottleneck = high_p[it]
                        break
                it -= 1
            if bottleneck == 0:
                bottleneck = close_p[it]
                             
        if display is True:
            predict_item(i, bottleneck, close_p[-1])
            
        d = round(((bottleneck-close_p[-1])/close_p[-1])*100.0,2)

        if d <= 0:
            fullfill.append(i)
            print_info['done'] += 1
            continue       
        elif d < 2:
            print_info['<2%'] += 1
        elif d >= 2 and d < 5:
            print_info['2%~5%'] += 1
        elif d >= 5 and d < 8:
            print_info['5%~8%'] += 1
        elif d >= 8:
            print_info['>8%'] += 1
        info[i] = d
    value = print_info['>8%']*1e4+print_info['5%~8%']*1e3+print_info['2%~5%']*1e2+print_info['<2%']*1e1+print_info['done']
    return fullfill, info, print_info, value

def getReserve_Remove(digit_time):
        f = pd.read_csv('data/2330.csv',sep='\n').values

        dt = np.array(["%d%s%s"%(int(s[0].split(",")[0].split("/")[0])+1911,s[0].split(",")[0].split("/")[1].zfill(2), s[0].split(",")[0].split("/")[2].zfill(2)) for s in f])

        idx = np.where(dt == str(digit_time))[0][0]
        dt = dt[idx+1-5:idx+1]

	remove_list = []  
        reserve_list = []      
	for e, dtt in enumerate(dt[::-1]):
	    if os.path.isfile('H.I./hide_'+dtt+'.txt') is True:

                for s in open('H.I./hide_'+dtt+'.txt', 'r').readlines():
                    if s == '\n':
                        continue

                    stock, d = s.split(",")[0], int(s.rstrip("\n").split(",")[1])

                    if d > e and stock not in remove_list:
                        remove_list += [stock]
                    if d == e:
                        reserve_list += [stock]
		    
	

	if os.path.isfile('H.I./reserve_'+dt[-2]+'.txt') is True:    
	    reserve_list += [str(int(s)) for s in np.loadtxt('H.I./reserve_'+dt[-2]+'.txt')] 

        return remove_list, reserve_list

