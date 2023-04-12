import requests
import json
import pandas as pd
import time
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pickle
import random

from lightfm import LightFM

from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import reciprocal_rank
from lightfm.evaluation import auc_score

from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split

np.random.seed(1)


def calc_weighted_rating(v, R, m, C):
    """
    v - количество оценок песни 
    m - необходимый минимум оценок для включения в список 
    R - средний рейтинг песни 
    C - среднее по всему списку оценок
    """
    return (v/(v+m) * R) + (m/(m+v) * C)

def get_n_popular_recommendations(ndf, smap, n=5):
    """
    Calculation of universal recommendation for our dataset
    """
    res = {}
    m = 30
    c = ndf['rating'].mean()
    play_count_dict = dict(ndf['song_id'].value_counts())
    average_rating = dict(ndf.groupby(by=['song_id'])['rating'].mean())
    for ave in average_rating:
        try:
            score = calc_weighted_rating(play_count_dict[ave], average_rating[ave], m, c)
        except Exception:
            print('Error occured')
        res[ave] = score
    r_list = list(dict(sorted(res.items(), key=lambda x:x[1], reverse=True)))

    result = []
    ids = r_list[:3 * n]
    for i in random.sample(ids, n):
        result.append(smap[i])
    return result

def get_what_user_heard(user_id, df, smap):
    result = []
    tracks = df.loc[df['user_id'] == user_id].sort_values(by='rating', ascending=True)['song_id']
    for t in tracks:
        result.append(smap[t])
    return result