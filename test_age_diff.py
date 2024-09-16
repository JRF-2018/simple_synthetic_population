#!/usr/bin/python3
__version__ = '0.0.1' # Time-stamp: <2024-09-13T21:38:31Z>
## Language: Japanese/UTF-8

import os
import sys
import re
import requests
import pandas as pd
import numpy as np
import random
from scipy.stats import norm
import matplotlib.pyplot as plt

DATA_DIR = './data'

## 令和２年国勢調査
## 表15-1
## 夫の年齢（5歳階級），夫婦のいる世帯の家族類型，子供の有無・数，最年少の子供の年齢，最年長の子供の年齢別一般世帯数（夫婦のいる一般世帯）－全国，都道府県，21大都市，特別区，人口50万以上の市
B15_01_URL = 'https://www.e-stat.go.jp/stat-search/file-download?statInfId=000032142522&fileKind=0'
B15_01_NAME = 'b15_01.xlsx'

## 表15-5
## 妻の年齢（5歳階級），夫婦のいる世帯の家族類型，子供の有無・数，最年少の子供の年齢，最年長の子供の年齢別一般世帯数（夫婦のいる一般世帯）－全国，都道府県，21大都市，特別区，人口50万以上の市
B15_05_URL = 'https://www.e-stat.go.jp/stat-search/file-download?statInfId=000032142526&fileKind=0'
B15_05_NAME = 'b15_05.xlsx'

## 表17-1
## 夫の年齢（各歳），妻の年齢（各歳）別夫婦数（一般世帯）－全国，都道府県，21大都市，21大都市の区，県庁所在市，人口20万以上の市
B17_01_URL = 'https://www.e-stat.go.jp/stat-search/file-download?statInfId=000032142538&fileKind=0'
B17_01_NAME = 'b17_01.xlsx'


def download_data_if_not_exists ():
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    url = B15_01_URL
    fn = os.path.join(DATA_DIR, B15_01_NAME)
    if not os.path.isfile(fn):
        c = requests.get(url).content
        with open(fn, mode='wb') as f:
            f.write(c)

    url = B15_05_URL
    fn = os.path.join(DATA_DIR, B15_05_NAME)
    if not os.path.isfile(fn):
        c = requests.get(url).content
        with open(fn, mode='wb') as f:
            f.write(c)
            
    url = B17_01_URL
    fn = os.path.join(DATA_DIR, B17_01_NAME)
    if not os.path.isfile(fn):
        c = requests.get(url).content
        with open(fn, mode='wb') as f:
            f.write(c)


def pickup_B15 (df, parent_age, num_child, min_age, max_age):
    row = df.index.get_loc(('a', '00000_全国', parent_age, num_child, min_age, max_age))
    q = df.iloc[row, 0]
    if isinstance(q, (int, np.integer)):
        return q
    else:
        return 0


def draw_B15 (B15_xlsx, ax, axb):
    df = pd.read_excel(B15_xlsx, header=list(range(6,10)),
                       index_col=list(range(0, 6)))
    data = [None] * 11
    for i in range(1, 12):
        if i == 11:
            l1 = '11_65歳以上'
            page = 65
        else:
            l1 = '%02d_%d～%d歳' % (i, (i - 1) * 5 + 15, (i - 1) * 5 + 19)
            page = (i - 1) * 5 + 15 + 2.5
        acc = []
        data[i-1] = acc
        for j in range(1, 5):
            if j == 4:
                l2 = '24_子供が4人以上'
            else:
                l2 = '2%d_子供が%d人' % (j, j)
            for k in range(1, 9):
                if k == 1:
                    min_age = 0.5
                    l3 = '1_0歳'
                elif k == 2:
                    min_age = 2.0
                    l3 = '2_1～2歳'
                elif k == 8:
                    min_age = 18
                    l3 = '8_18歳以上'
                else:
                    min_age = 3 + (k - 3) * 3 + 1.5
                    l3 = '%d_%d～%d歳' % (k, 3 + (k - 3) * 3,
                                          3 + (k - 3) * 3 + 2)
                if j == 1:
                    l4 = '0_総数'
                    max_age = min_age
                    q = pickup_B15(df, l1, l2, l3, l4)
                    acc.extend([page - ((max_age + min_age) / 2)] * q)
                else:
                    for l in range(1, 9):
                        if l == 1:
                            max_age = 0.5
                            l4 = '1_0歳'
                        elif l == 2:
                            max_age = 2.0
                            l4 = '2_1～2歳'
                        elif l == 8:
                            max_age = 18
                            l4 = '8_18歳以上'
                        else:
                            max_age = 3 + (l - 3) * 3 + 1.5
                            l4 = '%d_%d～%d歳' % (l, 3 + (l - 3) * 3,
                                                  3 + (l - 3) * 3 + 2)
                        q = pickup_B15(df, l1, l2, l3, l4)
                        acc.extend([page - ((max_age + min_age) / 2)] * j * q)

    ## 平均と標準偏差を求めるときは、50歳〜54歳以下に限定する。18歳以
    ## 上の子が計算に多くなると結果がゆがむので。
    dall = sum(data[0:8], [])
    mn = np.mean(dall)
    std = np.std(dall)
    print(mn, std)
    axb.hist(dall, bins=120, alpha=0.6, range=(0, 60), density=True)
    for i in range(len(data)):
        ax.hist(data[i], bins=120, alpha=0.6, range=(0, 60), density=True)
    x = np.linspace(0, 60, 100)
    p = norm.pdf(x, loc=mn, scale=std)
    ax.plot(x, p, 'k', linewidth=1)
    axb.plot(x, p, 'k', linewidth=1)


def pickup_B17_01 (df, male_age, female_age):
    row = male_age - 15 + 1
    col = female_age - 15 + 1
    q = df.iloc[row, col]
    if isinstance(q, (int, np.integer)):
        return q
    else:
        return 0


def draw_B17_01 (B17_01_xlsx, ax):
    df = pd.read_excel(B17_01_xlsx, header=list(range(7,10)),
                       index_col=list(range(0, 4)))
    data = []
    for i in range(15, 85):
        for j in range(15, 85):
            q = pickup_B17_01(df, i, j)
            data.extend([i - j] * q)

    mn = np.mean(data)
    std = np.std(data)
    print(mn, std)
    ax.hist(data, bins=120, alpha=0.6, range=(-20, 20), density=True)
    x = np.linspace(-20, 20, 100)
    p = norm.pdf(x, loc=mn, scale=std)
    ax.plot(x, p, 'k', linewidth=1)


def test_plot ():
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    #ax6 = fig.add_subplot(3, 2, 6)

    draw_B15(os.path.join(DATA_DIR, B15_01_NAME), ax1, ax2)
    draw_B15(os.path.join(DATA_DIR, B15_05_NAME), ax3, ax4)
    draw_B17_01(os.path.join(DATA_DIR, B17_01_NAME), ax5)
    plt.show()


if __name__ == '__main__':
    download_data_if_not_exists()
    test_plot()

## 結果
# 32.757869148885014 4.829387554625909 #父子年齢差
# 31.512137597064076 4.349459715779061 #母子年齢差
# 2.308170609587958 4.110029017891528  #夫婦年齢差
