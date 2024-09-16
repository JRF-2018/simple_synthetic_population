#!/usr/bin/python3
__version__ = '0.0.1' # Time-stamp: <2024-09-15T16:21:33Z>
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
## 表11-1
## 世帯主の男女，世帯主の年齢（5歳階級），世帯の家族類型，世帯人員の人数別一般世帯数－全国
B11_01_URL = 'https://www.e-stat.go.jp/stat-search/file-download?statInfId=000032142502&fileKind=0'
B11_01_NAME = 'b11_01.xlsx'

## 表11-2
## 世帯主の男女，世帯主の年齢（5歳階級），世帯の家族類型，世帯人員の人数別一般世帯人員－全国
B11_02_URL = 'https://www.e-stat.go.jp/stat-search/file-download?statInfId=000032142503&fileKind=0'
B11_02_NAME = 'b11_02.xlsx'


def download_data_if_not_exists ():
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    url = B11_01_URL
    fn = os.path.join(DATA_DIR, B11_01_NAME)
    if not os.path.isfile(fn):
        c = requests.get(url).content
        with open(fn, mode='wb') as f:
            f.write(c)

    url = B11_02_URL
    fn = os.path.join(DATA_DIR, B11_02_NAME)
    if not os.path.isfile(fn):
        c = requests.get(url).content
        with open(fn, mode='wb') as f:
            f.write(c)

## 家族類型
B11_CODE = [None] * 17
##  核家族世帯
B11_CODE[ 0] = '111_夫婦のみの世帯'
B11_CODE[ 1] = '112_夫婦と子供から成る世帯'
B11_CODE[ 2] = '113_男親と子供から成る世帯'
B11_CODE[ 3] = '114_女親と子供から成る世帯'
##  核家族以外の世帯
B11_CODE[ 4] = '1201_夫婦と両親から成る世帯'
B11_CODE[ 5] = '1202_夫婦とひとり親から成る世帯'
B11_CODE[ 6] = '1203_夫婦，子供と両親から成る世帯'
B11_CODE[ 7] = '1204_夫婦，子供とひとり親から成る世帯'
B11_CODE[ 8] = '1205_夫婦と他の親族（親，子供を含まない）から成る世帯'
B11_CODE[ 9] = '1206_夫婦，子供と他の親族（親を含まない）から成る世帯'
B11_CODE[10] = '1207_夫婦，親と他の親族（子供を含まない）から成る世帯'
B11_CODE[11] = '1208_夫婦，子供，親と他の親族から成る世帯'
B11_CODE[12] = '1209_兄弟姉妹のみから成る世帯'
B11_CODE[13] = '1210_他に分類されない世帯'
##  その他
B11_CODE[14] = '2_非親族を含む世帯'
B11_CODE[15] = '3_単独世帯'
B11_CODE[16] = '4_世帯の家族類型「不詳」'

## 世帯主の男女
## 0: 1_男
## 1: 2_女

##世帯主の年齢
##  0: 01_15歳未満
##  1: 02_15～19歳
## ...
## 14: 15_80～84歳
## 15: 16_85歳以上

## 世帯人員の人数
## 1_世帯人員が1人
## ...
## 7_世帯人員が7人以上

B11_data1 = None
B11_over7num = None
B11_alist = None

def pickup_B11 (df, l1, l2, l3, l4):
    row = df.index.get_loc(('00000_全国', l1, l2, l3))
    for k in df.columns:
        if (k[0] == l4):
            col = df.columns.get_loc(k)
            break
    q = df.iloc[row, col]
    if isinstance(q, (int, np.integer)):
        return q
    else:
        return 0


def load_B11 (B11_xlsx):
    df = pd.read_excel(B11_xlsx, header=list(range(7,10)),
                       index_col=list(range(0, 4)))
    data = [None] * 2
    for sex in range(2):
        data[sex] = [None] * 16
        if sex == 0:
            l1 = '1_男'
        else: 
            l1 = '2_女'
        for age in range(16):
            data[sex][age] = [None] * 7
            if age == 15:
                l2 = '16_85歳以上'
            elif age == 0:
                l2 = '01_15歳未満'
            else:
                l2 = '%02d_%d～%d歳' % (age + 1, (age - 1) * 5 + 15,
                                        (age - 1) * 5 + 19)
            for num in range(7):
                data[sex][age][num] = [0] * 17
                if num == 6:
                    l3 = '7_世帯人員が7人以上'
                else:
                    l3 = '%d_世帯人員が%d人' % (num + 1, num + 1)
                for code in range(17):
                    q = pickup_B11(df, l1, l2, l3, B11_CODE[code])
                    data[sex][age][num][code] = q
    return data


def initialize_B11 (B11_01_xlsx, B11_02_xlsx):
    global B11_data1, B11_over7num, B11_alist
    
    data1 = load_B11(B11_01_xlsx)
    data2 = load_B11(B11_02_xlsx)

    B11_data1 = data1

    over7num = [None] * 2
    for sex in range(2):
        over7num[sex] = [None] * 16
        for age in range(16):
            over7num[sex][age] = [0] * 17
            for code in range(17):
                p = data1[sex][age][6][code]
                q = data2[sex][age][6][code]
                if p != 0:
                    r = q / p
                    assert r >= 7
                    over7num[sex][age][code] = r - 7
    B11_over7num = over7num
    alist = [None] * 2
    acc = 0
    for sex in range(2):
        alist[sex] = [None] * 16
        for age in range(16):
            alist[sex][age] = [None] * 7
            for num in range(7):
                alist[sex][age][num] = [None] * 17
                for code in range(17):
                    q = data1[sex][age][num][code]
                    acc += q
                    alist[sex][age][num][code] = acc
    B11_alist = alist
    


## sex, age, code についてはコードが返るが、num についてはコードではな
## く実数が返るのを間違えなきよう。
def rand_family ():
    q = random.uniform(0, B11_alist[1][15][6][16])

    for sex in range(2):
        if q < B11_alist[sex][15][6][16]:
            break
    for age in range(16):
        if q < B11_alist[sex][age][6][16]:
            break
    for num in range(7):
        if q < B11_alist[sex][age][num][16]:
            break
    for code in range(17):
        if q < B11_alist[sex][age][num][code]:
            break
    num = num + 1
    if num == 7:
        p = B11_over7num[sex][age][code]
        num += np.random.poisson(p)
    return sex, age, num, code


def age_to_agecode (age):
    if age < 15:
        return 0
    elif age >= 85:
        return 15
    else:
        return int((age - 15) / 5) + 1

def rand_parents (sex=None, age=None, has_children=None):
    dbsum = 0
    sgsum = 0
    asex = sex
    aage = age
    if asex is not None:
        asex = 0 if asex == 'M' else 1
    if aage is not None:
        aage = age_to_agecode(aage)

    for sex in range(2):
        if asex is not None:
            if asex != sex:
                continue
        for age in range(16):
            if aage is not None:
                if aage != age:
                    continue
            for num in range(7):
                for code in range(17):
                    if has_children is not None:
                        if has_children:
                            if code not in [6, 7]:
                                continue
                        else:
                            if code not in [4, 5]:
                                continue
                    if code in [4, 6]:
                        dbsum += B11_data1[sex][age][num][code]
                    elif code in [5, 7]:
                        sgsum += B11_data1[sex][age][num][code]

    return random.random() < dbsum / (dbsum + sgsum)


if __name__ == '__main__':
    download_data_if_not_exists()
    initialize_B11(os.path.join(DATA_DIR, B11_01_NAME),
                   os.path.join(DATA_DIR, B11_02_NAME))

    print(rand_parents())
    print(rand_parents())
    print(rand_parents())
    print(rand_parents(sex='M', age=18, has_children=True))
    print(rand_parents(sex='M', age=18, has_children=True))
    print(rand_parents(sex='M', age=18, has_children=True))

    sex, age, num, code = rand_family()
    print(sex, age, num, code)
    sex, age, num, code = rand_family()
    print(sex, age, num, code)
    sex, age, num, code = rand_family()
    print(sex, age, num, code)
    sex, age, num, code = rand_family()
    print(sex, age, num, code)
    sex, age, num, code = rand_family()
    print(sex, age, num, code)

    while True:
        sex, age, num, code = rand_family()
        print(sex, age, num, code)
        if num > 7:
            break
