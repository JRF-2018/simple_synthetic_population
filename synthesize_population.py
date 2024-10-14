#!/usr/bin/python3
__version__ = '0.0.2' # Time-stamp: <2024-10-13T22:21:45Z>
## Language: Japanese/UTF-8

import argparse
ARGS = argparse.Namespace()

ARGS.data_dir = './data'
ARGS.output = 'population.csv'
ARGS.population = 1000
ARGS.t0 = 1000 		# 初期温度
ARGS.alpha = 0.95	# 冷却率
ARGS.beta = 1           # 定数
ARGS.m = 100            # 次のパラメータ更新までの時間
ARGS.max_time = 100000  # アニーリング過程に与えられた総処理時間
ARGS.wasserstein = False # True: スコアにワッサーシュタイン距離を使う。
                         # False: スコアに KL ダイバージェンスを使う。
ARGS.swap_hack = True   # True: 近傍を求めるとき親と子の年齢をチェックする。
                        # False: 近傍を求めるとき乱数に頼る。
ARGS.reproduct_hack = False # True: 初期生成時に子や親の年齢書き換えを許す。
                            # False: 許さない。
ARGS.family_cond_hack = True # True: 初期に家族条件を強制。
                             # False: 強制しない。
ARGS.check_12 = True    # True: 子のいる世帯主は12歳以上にする。
                        # False: しない。
ARGS.check_95 = True    # True: 親のいる世帯主は95歳未満にする。
                        # False: しない。


## test_age_diff.py の結果
ARGS.fa_c_mean = 32.757869148885014 # 父子年齢差
ARGS.fa_c_std = 4.829387554625909
ARGS.mo_c_mean = 31.512137597064076 # 母子年齢差
ARGS.mo_c_std = 4.349459715779061
ARGS.m_f_mean = 2.308170609587958   # 夫婦年齢差
ARGS.m_f_std = 4.110029017891528


import os
import sys
import re
import requests
import pandas as pd
import numpy as np
import random
import copy
import csv
from scipy.stats import norm
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

from rand_age_sex import rand_age, rand_sex, initialize_B02_01, \
    B02_01_URL, B02_01_NAME
from rand_family import rand_family, initialize_B11, \
    age_to_agecode, rand_parents, \
    B11_01_URL, B11_01_NAME, B11_02_URL, B11_02_NAME


EPS = np.finfo(float).eps

class Person:
    def __init__ (self):
        self.id = None         # 個人 ID
        self.sex = None        # 性別 'M' or 'F'
        self.age = None        # 年齢

class Family:
    def __init__ (self):
        self.id = None         # 世帯 ID
        self.code = None       # 世帯種コード
        self.num = None        # 世帯人数
        self.master = None     # 世帯主
        self.spouse = None     # 配偶者
        self.parents = []      # 親
        self.children = []     # 子供
        self.others = []       # 配偶者と親と子供以外

    def clone (self):
        c = copy.copy(self)
        c.parents = [p for p in c.parents]
        c.children = [p for p in c.children]
        c.others = [p for p in c.others]
        return c


def parse_args (view_options=['none']):
    parser = argparse.ArgumentParser()

    # parser.add_argument("-p", "--population", type=str)

    specials = set()
    #specials = set(['load', 'save', 'debug_on_error', 'debug_term',
    #                'trials', 'population', 'min_birth',
    #                'view_1', 'view_2', 'view_3', 'view_4'])
    for p, v in vars(ARGS).items():
        if p not in specials:
            p2 = '--' + p.replace('_', '-')
            np2 = '--no-' + p.replace('_', '-')
            if np2.startswith('--no-no-'):
                np2 = np2.replace('--no-no-', '--with-', 1)
            if v is False or v is True:
                parser.add_argument(p2, action="store_true")
                parser.add_argument(np2, action="store_false", dest=p)
            elif v is None:
                parser.add_argument(p2, type=float)
            else:
                parser.add_argument(p2, type=type(v))
    
    parser.parse_args(namespace=ARGS)


def download_data_if_not_exists ():
    if not os.path.isdir(ARGS.data_dir):
        os.mkdir(ARGS.data_dir)

    url = B02_01_URL
    fn = os.path.join(ARGS.data_dir, B02_01_NAME)
    if not os.path.isfile(fn):
        c = requests.get(url).content
        with open(fn, mode='wb') as f:
            f.write(c)

    url = B11_01_URL
    fn = os.path.join(ARGS.data_dir, B11_01_NAME)
    if not os.path.isfile(fn):
        c = requests.get(url).content
        with open(fn, mode='wb') as f:
            f.write(c)

    url = B11_02_URL
    fn = os.path.join(ARGS.data_dir, B11_02_NAME)
    if not os.path.isfile(fn):
        c = requests.get(url).content
        with open(fn, mode='wb') as f:
            f.write(c)


def serialize_family (f):
    s = [['M', f.master]]
    if f.spouse:
        s.append(['S', f.spouse])
    s.extend([['P', p] for p in f.parents])
    s.extend([['C', p] for p in f.children])
    s.extend([['O', p] for p in f.others])
    return s
    
def output_population (filename, people, families):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for f1 in families:
            for a, p in serialize_family(f1):
                writer.writerow(['F%05d' % f1.id,
                                 f1.code,
                                 f1.num,
                                 a,
                                 'P%05d' % p.id,
                                 p.sex,
                                 p.age])

def rand_age_list (l):
    sz = sum([len(l[i]) for i in range(len(l))])
    if sz == 0:
        return None
    k = random.randrange(sz)
    acc = 0
    for l1 in l:
        acc += len(l1)
        if k < acc:
            k = k - (acc - len(l1))
            q = l1.pop(k)
            break
    return q
    

def rand_male (age_people):
    return rand_age_list(age_people[0])

def rand_female (age_people):
    return rand_age_list(age_people[1])

def rand_someone (age_people):
    return rand_age_list(age_people[0] + age_people[1])

def rand_child (age_people, age):
    cto = age_to_agecode(age - 13)
    q = rand_age_list(age_people[0][0:cto+1] + age_people[1][0:cto+1])
    if q is None:
        q = rand_someone(age_people)
        if q is not None:
            if ARGS.reproduct_hack:
                q.age = rand_age(sex=q.sex, max=age - 13)
    return q

def rand_parent (age_people, age):
    cfrom = age_to_agecode(age + 13)
    q = rand_age_list(age_people[0][cfrom:] + age_people[1][cfrom:])
    if q is None:
        q = rand_someone(age_people)
        if q is not None:
            if ARGS.reproduct_hack:
                q.age = rand_age(sex=q.sex, min=age + 13)
    return q

def rand_male_parent (age_people, age):
    cfrom = age_to_agecode(age + 13)
    q = rand_age_list(age_people[0][cfrom:])
    if q is None:
        q = rand_male(age_people)
        if q is not None:
            if ARGS.reproduct_hack:
                q.age = rand_age(sex=q.sex, min=age + 13)
    return q

def rand_female_parent (age_people, age):
    cfrom = age_to_agecode(age + 13)
    q = rand_age_list(age_people[1][cfrom:])
    if q is None:
        q = rand_female(age_people)
        if q is not None:
            if ARGS.reproduct_hack:
                q.age = rand_age(sex=q.sex, min=age + 13)
    return q

def rand_male_spouse (age_people, age):
    cfrom = age_to_agecode(age - 10)
    cto = age_to_agecode(age + 10)
    q = rand_age_list(age_people[0][cfrom:cto+1])
    if q is None:
        q = rand_male(age_people)
        if q is not None:
            if ARGS.reproduct_hack:
                q.age = rand_age(sex=q.sex, min=age - 13, max=age +13)
    return q

def rand_female_spouse (age_people, age):
    cfrom = age_to_agecode(age - 10)
    cto = age_to_agecode(age + 10)
    q = rand_age_list(age_people[1][cfrom:cto+1])
    if q is None:
        q = rand_female(age_people)
        if q is not None:
            if ARGS.reproduct_hack:
                q.age = rand_age(sex=q.sex, min=age - 13, max=age +13)
    return q


def unpop_person (age_people, p):
    if p is None:
        return
    sex = 0 if p.sex == 'M' else 1
    acode = age_to_agecode(p.age)
    age_people[sex][acode].append(p)

def unpop_family (age_people, f):
    if f.others:
        for p in f.others:
            unpop_person(age_people, p)
    if f.children:
        for p in f.children:
            unpop_person(age_people, p)
    if f.parents:
        for p in f.parents:
            unpop_person(age_people, p)
    if f.spouse:
        unpop_person(age_people, f.spouse)
    unpop_person(age_people, f.master)

def make_half_family(age_people, f, has_spouse, has_parents,
                     has_children, has_others):
    done = True
    r = f.num - 1
    if has_spouse:
        if f.master.sex == 'M':
            q = rand_female_spouse(age_people, f.master.age)
            if q is None:
                done = False
            f.spouse = q
            r -= 1
        else:
            q = rand_male_spouse(age_people, f.master.age)
            if q is None:
                done = False
            f.spouse = q
            r -= 1

    if has_parents == 1.5:
        if rand_parents(f.master.sex, f.master.age, has_children) \
           and r - 2 >= int(has_children) + int(has_others):
            has_parents = 2
        else:
            has_parents = 1
    if has_parents == 1:
        q = rand_parent(age_people, f.master.age)
        if q is None:
            done = False
        f.parents.append(q)
        r -= 1
    elif has_parents == 2:
        q = rand_male_parent(age_people, f.master.age)
        if q is None:
            done = False
        f.parents.append(q)
        q = rand_female_parent(age_people, f.master.age)
        if q is None:
            done = False
        f.parents.append(q)
        r -= 2

    return done

def make_complete_family(age_people, f, has_spouse, has_parents,
                         has_children, has_others):
    done = True
    r = f.num - 1
    if has_spouse:
        r -= 1
    if f.parents:
        r -= len(f.parents)

    if has_children:
        if has_others:
            cn = random.randrange(1, r - 1 + 1)
        else:
            cn = r
        for i in range(cn):
            q = rand_child(age_people, f.master.age)
            if q is None:
                done = False
                break
            f.children.append(q)
        r -= cn

    if has_others:
        for i in range(r):
            q = rand_someone(age_people)
            if q is None:
                done = False
                break
            f.others.append(q)
        r = 0

    assert r == 0
    return done


def initial_step (population):
    people = [None] * population
    for i in range(population):
        p = Person()
        p.id = i
        p.sex = rand_sex()
        p.age = rand_age(p.sex)
        people[i] = p

    age_people = [None] * 2
    age_people[0] = [[] for i in range(16)]
    age_people[1] = [[] for i in range(16)]
    for p in people:
        unpop_person(age_people, p)

    families = []
    acc = 0
    prev_print = None
    while acc < population:
        sex, age, num, code = rand_family()

        if not age_people[sex][age]:
            continue

        f = Family()
        f.id = len(families)
        p = age_people[sex][age].pop(0)
        f.code = code
        f.num = num

        if ARGS.check_12 and code in [1, 2, 3, 6, 7, 9, 11] and p.age < 12:
            k = len(age_people[sex][age]) + 1
            while p.age < 12 and k > 0:
                age_people[sex][age].append(p)
                p = age_people[sex][age].pop(0)
                k = k - 1
            if p.age < 12:
                continue

        if ARGS.check_95 and code in [4, 5, 6, 7, 10, 11] and p.age >= 95:
            k = len(age_people[sex][age]) + 1
            while p.age >= 95 and k > 0:
                age_people[sex][age].append(p)
                p = age_people[sex][age].pop(0)
                k = k - 1
            if p.age >= 95:
                continue

        f.master = p
        done = True
        if f.code == 0:
            done = make_half_family(age_people, f, True, 0, False, False)
        elif f.code == 1:
            done = make_half_family(age_people, f, True, 0, True, False)
        elif f.code == 2 or f.code == 3:
            done = make_half_family(age_people, f, False, 0, True, False)
        elif f.code == 4:
            done = make_half_family(age_people, f, True, 2, False, False)
        elif f.code == 5:
            done = make_half_family(age_people, f, True, 1, False, False)
        elif f.code == 6:
            done = make_half_family(age_people, f, True, 2, True, False)
        elif f.code == 7:
            done = make_half_family(age_people, f, True, 1, True, False)
        elif f.code == 8:
            done = make_half_family(age_people, f, True, 0, False, True)
        elif f.code == 9:
            done = make_half_family(age_people, f, True, 0, True, True)
        elif f.code == 10:
            done = make_half_family(age_people, f, True, 1.5, False, True)
        elif f.code == 11:
            done = make_half_family(age_people, f, True, 1.5, True, True)
        elif f.code == 12:
            done = make_half_family(age_people, f, False, 0, False, True)
        elif f.code == 13:
            done = False
        elif f.code == 14:
            done = False
        elif f.code == 15:
            pass
        elif f.code == 16:
            done = False
        if done and acc + f.num <= population:
            acc += f.num
            families.append(f)
            if (acc % 100 == 0 and population - acc < 1000):
                print(len(families), acc)
        else:
            unpop_family(age_people, f)

    for f in families:
        if f.code == 0:
            done = make_complete_family(age_people, f, True, 0, False, False)
        elif f.code == 1:
            done = make_complete_family(age_people, f, True, 0, True, False)
        elif f.code == 2 or f.code == 3:
            done = make_complete_family(age_people, f, False, 0, True, False)
        elif f.code == 4:
            done = make_complete_family(age_people, f, True, 2, False, False)
        elif f.code == 5:
            done = make_complete_family(age_people, f, True, 1, False, False)
        elif f.code == 6:
            done = make_complete_family(age_people, f, True, 2, True, False)
        elif f.code == 7:
            done = make_complete_family(age_people, f, True, 1, True, False)
        elif f.code == 8:
            done = make_complete_family(age_people, f, True, 0, False, True)
        elif f.code == 9:
            done = make_complete_family(age_people, f, True, 0, True, True)
        elif f.code == 10:
            done = make_complete_family(age_people, f, True, 1.5, False, True)
        elif f.code == 11:
            done = make_complete_family(age_people, f, True, 1.5, True, True)
        elif f.code == 12:
            done = make_complete_family(age_people, f, False, 0, False, True)
        elif f.code == 13:
            done = True
        elif f.code == 14:
            done = True
        elif f.code == 15:
            pass
        elif f.code == 16:
            done = True
        if not done:
            print("error!:", f.code, f.num, f.children, f.parents, len(sum(age_people[0], [])), len(sum(age_people[1], [])))
            sys.exit(1)

    return people, families


def unserialize_family(f, s):
    f.master = s[0][1]
    i = 1
    if f.spouse:
        f.spouse = s[i][1]
        i += 1
    f.parents = [p for m, p in s[i:i+len(f.parents)]]
    i += len(f.parents)
    f.children = [p for m, p in s[i:i+len(f.children)]]
    i += len(f.children)
    f.others = [p for m, p in s[i:i+len(f.others)]]
    i += len(f.others)


def check_family_cond(master, attr, member):
    if attr == 'C':
        if master.age <= member.age:
            return False
    if attr == 'P':
        if master.age >= member.age:
            return False
    if attr == 'S':
        if member.age < 16:
            return False
    return True


def check_family_cond_full(serialized_family):
    f = serialized_family
    m = f[0][1]
    for attr, member in serialized_family:
        if not check_family_cond(m, attr, member):
            return False
    return True


def enforce_family_cond(families):
    fnc = []
    fc = []
    for i, f in enumerate(families):
        sf = serialize_family(f)
        if check_family_cond_full(sf):
            fc.append([i, sf])
        else:
            fnc.append([i, sf])

    while fnc:
        i1 = 0
        s1 = fnc[i1][1]
        j1 = 0
        for j1 in range(1, len(s1)):
            if not check_family_cond(s1[0][1], s1[j1][0], s1[j1][1]):
                break
        done = False
        i2 = 0
        for i2 in range(1, len(fnc) + len(fc)):
            s2 = fnc[i2][1] if i2 < len(fnc) else fc[i2 - len(fnc)][1]
            if len(s2) == 1:
                continue
            for j2 in range(1, len(s2)):
                j2 = random.randrange(1, len(s2))
                if s1[j1][1].sex != s2[j2][1].sex:
                    continue
                if not check_family_cond(s1[0][1], s1[j1][0], s2[j2][1]):
                    continue
                if not check_family_cond(s2[0][1], s2[j2][0], s1[j1][1]):
                    continue
                s1[j1][1], s2[j2][1] = s2[j2][1], s1[j1][1]
                if i2 < len(fnc) and check_family_cond_full(s2):
                    fc.append(fnc.pop(i2))
                if check_family_cond_full(s1):
                    fc.append(fnc.pop(0))
                done = True
                break
            if done:
                break
        if not done:
            raise ValueError("failed: enforce_family_cond.")

    for i, sf in fc:
        unserialize_family(families[i], sf)
    return families


def anealing_neighbor(families):
    r = [f for f in families]
    while True:
        i1 = random.randrange(len(r))
        i2 = random.randrange(len(r))
        if i1 == i2:
            continue
        f1 = r[i1].clone()
        f2 = r[i2].clone()
        if f1.num == 1 or f2.num == 1:
            continue
        j1 = random.randrange(1, f1.num)
        j2 = random.randrange(1, f2.num)
        s1 = serialize_family(f1)
        s2 = serialize_family(f2)
        if s1[j1][1].sex != s2[j2][1].sex:
            continue
        if ARGS.swap_hack:
            if not check_family_cond(f1.master, s1[j1][0], s2[j2][1]):
                continue
            if not check_family_cond(f2.master, s2[j2][0], s1[j1][1]):
                continue
        s1[j1][1], s2[j2][1] = s2[j2][1], s1[j1][1]
        unserialize_family(f1, s1)
        unserialize_family(f2, s2)
        r[i1] = f1
        r[i2] = f2
        break
    return r


def diff_family (f):
    diff_fa_c = []
    diff_mo_c = []
    diff_m_f = []

    father = None
    mother = None
    grand_father = None
    grand_mother = None

    if f.master.sex == 'M':
        father = f.master
        if f.spouse:
            mother = f.spouse
    else:
        mother = f.master
        if f.spouse:
            father = f.spouse

    if len(f.parents) == 2:
        if f.parents[0].sex == 'M':
            grand_father = f.parents[0]
            grand_mother = f.parents[1]
        else:
            grand_father = f.parents[1]
            grand_mother = f.parents[0]
    if len(f.parents) == 2:
        if f.parents[0].sex == 'M':
            grand_father = f.parents[0]
        else:
            grand_mother = f.parents[0]

    if father and mother:
        diff_m_f.append(father.age - mother.age)
    if grand_father and grand_mother:
        diff_m_f.append(grand_father.age - grand_mother.age)

    if father and f.children:
        for c in f.children:
            diff_fa_c.append(father.age - c.age)
    
    if mother and f.children:
        for c in f.children:
            diff_mo_c.append(mother.age - c.age)

    if grand_father and father:
        diff_fa_c.append(grand_father.age - father.age)

    if grand_father and mother:
        diff_fa_c.append(grand_father.age - mother.age)
        
    if grand_mother and father:
        diff_mo_c.append(grand_mother.age - father.age)

    if grand_mother and mother:
        diff_mo_c.append(grand_mother.age - mother.age)

    return diff_fa_c, diff_mo_c, diff_m_f


def kl_divergence(p, q):
    # KLダイバージェンスの計算
    kl = np.sum(p * np.log(p / q))
    return kl

def kl_divergence_alpha(p, q, alpha=1.0):
    # ラプラススムージング
    p = (p + alpha) / (np.sum(p) + alpha * p.size)
    q = (q + alpha) / (np.sum(q) + alpha * q.size)

    return kl_divergence(p, q)

def kl_divergence_eps(p, q, eps=EPS):
    # 平滑化
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)

    return kl_divergence(p, q)

## データとある標準偏差と平均を持つ正規分布を比較
def data_normal_kl_divergence (data, mean, std, alpha=0.01, bins=100):
    # ヒストグラムの作成
    counts, bins = np.histogram(data, bins=bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # 正規分布の確率密度関数
    q = norm.pdf(bin_centers, loc=mean, scale=std)

    #return kl_divergence_alpha(counts, q, alpha)
    return kl_divergence_eps(counts, q)


def data_normal_wasserstein(data, mean, std):
    # 正規分布からサンプルを生成
    normal_dist = np.random.normal(loc=mean, scale=std, size=len(data))

    # ワッサーシュタイン距離を計算
    wasserstein = wasserstein_distance(data, normal_dist)

    return wasserstein

def anealing_cost(families):
    diff_fa_c = []
    diff_mo_c = []
    diff_m_f = []
    
    for f in families:
        d1, d2, d3 = diff_family (f)
        diff_fa_c.extend(d1)
        diff_mo_c.extend(d2)
        diff_m_f.extend(d3)

    if ARGS.wasserstein:
        cost_fa_c = data_normal_wasserstein(diff_fa_c, ARGS.fa_c_mean,
                                            ARGS.fa_c_std)
        cost_mo_c = data_normal_wasserstein(diff_mo_c, ARGS.mo_c_mean,
                                            ARGS.mo_c_std)
        cost_m_f = data_normal_wasserstein(diff_m_f, ARGS.m_f_mean,
                                           ARGS.m_f_std)
    else:
        cost_fa_c = data_normal_kl_divergence(diff_fa_c, ARGS.fa_c_mean,
                                              ARGS.fa_c_std)
        cost_mo_c = data_normal_kl_divergence(diff_mo_c, ARGS.mo_c_mean,
                                              ARGS.mo_c_std)
        cost_m_f = data_normal_kl_divergence(diff_m_f, ARGS.m_f_mean,
                                             ARGS.m_f_std)
    return cost_fa_c + cost_mo_c + cost_m_f


def metropolis (CurS, CurCost, BestS, BestCost, T, M):
    while M > 0:
        NewS = anealing_neighbor(CurS)
        NewCost = anealing_cost(NewS)
        delta = NewCost - CurCost
        if delta < 0:
            CurS = NewS
            CurCost = NewCost
            if NewCost < BestCost:
                BestS = NewS
                BestCost = NewCost
        else:
            if random.random() < np.exp(- delta / T):
                CurS = NewS
                CurCost = NewCost
        M = M - 1
    return CurS, CurCost, BestS, BestCost


def simulated_anealing (S0, T0, ALPHA, BETA, M, MAX_TIME):
    # S0: 初期解
    T = T0
    CurS = S0
    BestS = CurS # BestSはここまでに得られた最良解
    CurCost = anealing_cost(CurS)
    BestCost = CurCost
    Time = 0
    while Time < MAX_TIME:
        CurS, CurCost, BestS, BestCost = \
            metropolis(CurS, CurCost, BestS, BestCost, T, M)
        Time = Time + M
        print(Time, BestCost, CurCost)
        T = ALPHA * T
        M = BETA * M
    return BestS, BestCost


if __name__ == '__main__':
    parse_args()
    download_data_if_not_exists()
    initialize_B02_01(os.path.join(ARGS.data_dir, B02_01_NAME))
    initialize_B11(os.path.join(ARGS.data_dir, B11_01_NAME),
                   os.path.join(ARGS.data_dir, B11_02_NAME))

    while True:
        people, families = initial_step(ARGS.population)
        print("done initial step.")
        if not ARGS.family_cond_hack:
            break
        try: 
            families = enforce_family_cond(families)
            print("enforced family cond.")
            break
        except ValueError:
            print("Failed: enforce_family_cond. Retrying...")

    families, _ = simulated_anealing(families, ARGS.t0, ARGS.alpha,
                                     ARGS.beta, ARGS.m, ARGS.max_time)
    output_population(ARGS.output, people, families)
