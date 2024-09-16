#!/usr/bin/python3
__version__ = '0.0.1' # Time-stamp: <2024-09-15T02:37:09Z>
## Language: Japanese/UTF-8

import os
import sys
import re
import requests
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

DATA_DIR = './data'

## 令和２年国勢調査
## 表2-1
## 男女，年齢（各歳），国籍総数か日本人別人口，平均年齢及び年齢中位数－全国，都道府県，21大都市，特別区，人口50万以上の市
B02_01_URL = 'https://www.e-stat.go.jp/stat-search/file-download?statInfId=000032142404&fileKind=0'
B02_01_NAME = 'b02_01.xlsx'

B02_01_Male = None
B02_01_Female = None


def download_data_if_not_exists ():
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    url = B02_01_URL
    fn = os.path.join(DATA_DIR, B02_01_NAME)
    if not os.path.isfile(fn):
        c = requests.get(url).content
        with open(fn, mode='wb') as f:
            f.write(c)
    

def initialize_B02_01 (B02_01_xlsx):
    global B02_01_Male, B02_01_Female

    df = pd.read_excel(B02_01_xlsx, header=list(range(7,11)),
                       index_col=list(range(0, 4)))

    m_index = None
    f_index = None
    for k in df.index:
        if k[0] == '0_国籍総数' and k[1] == '1_男' and k[3] == '00000_全国':
            m_index = df.index.get_loc(k)
        if k[0] == '0_国籍総数' and k[1] == '2_女' and k[3] == '00000_全国':
            f_index = df.index.get_loc(k)
    assert m_index
    assert f_index

    m_list = []
    f_list = []
    m_alist = []
    f_alist = []
    m_a = 0
    f_a = 0

    for age in range(0, 120):
        age_str = "\\d+_%d歳(?:以上)?" % age
        if re.match(age_str, df.columns[age + 1][1]):
            c = df.columns.get_loc(df.columns[age + 1])
            d = df.iloc[m_index, c]
            if isinstance(d, (np.integer, int)):
                m_list.append(d)
                m_a += d
                m_alist.append(m_a)
            else:
                m_list.append(0)
                m_alist.appaned(m_a)
            d = df.iloc[f_index, c]
            if isinstance(d, (np.integer, int)):
                f_list.append(d)
                f_a += d
                f_alist.append(f_a)
            else:
                f_list.append(0)
                f_alist.append(f_a)
        else:
            break
    
    B02_01_Male = m_alist
    B02_01_Female = f_alist


def rand_sex ():
    m_pop = B02_01_Male[-1]
    f_pop = B02_01_Female[-1]

    if random.uniform(0, m_pop + f_pop) < m_pop:
        return 'M'
    else:
        return 'F'


def rand_age (sex=None, min=None, max=None):
    if (sex is None):
        sex = rand_sex()

    assert sex == 'M' or sex == 'F'

    if sex == 'M':
        alist = B02_01_Male
    else:
        alist = B02_01_Female
    mn = 0
    mx = alist[-1]
    if min is not None:
        if min > 0:
            i = int(min)
            prev = 0
            if i - 1 >= 0:
                prev = alist[i - 1]
            cur = alist[i]
            mn = (min - i) * (cur - prev) + prev
    if max is not None:
        if max > 100:
            max = 100
        if max < 0:
            max = 0
        if max >= 0:
            i = int(max)
            prev = 0
            if i - 1 >= 0:
                prev = alist[i - 1]
            cur = alist[i]
            mx = (max - i) * (cur - prev) + prev
    k = random.uniform(mn, mx)
    for i in range(0, len(alist)):
        if k < alist[i]:
            q1 = 0
            q2 = alist[i]
            if i > 0:
                q1 = alist[i - 1]
            age = i + ((k - q1) / (q2 - q1))
            return age
    assert False


def test_plot ():
    pm = []
    pf = []
    for i in range(100000):
        sex = rand_sex()
        if sex == 'M':
            pm.append(rand_age(sex))
        else:
            pf.append(rand_age(sex))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.hist(pm, bins=120, alpha=0.6)
    ax.hist(pf, bins=120, alpha=0.6)
    ax.set_title('population')
    ax.set_xlabel('age')
    ax.set_ylabel('freq')
    plt.show()


if __name__ == '__main__':
    download_data_if_not_exists()
    initialize_B02_01(os.path.join(DATA_DIR, B02_01_NAME))
    test_plot()
