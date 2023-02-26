##  clean_dataset(df):
##  make_pivot(df):
##  calc_weighted_rating(v, R, m, C):

import utils

def clean_dataset(df):
    df['rank'] = df['rank'].astype(int)
    df['rating'] = (101 - df['rank'])/100
    df = df.loc[df['rating'] > 0.5].reset_index(drop=True)
    df['song_name'] = [str(x)[:40] for x in list(df['artist_song'])]
    df = df[['login', 'rating', 'song_name', 'artist', 'count']]
    df = df.drop_duplicates().reset_index(drop=True)
    ex_logins = list(df.loc[(df['rating'] > 0.5) & (df['count'] == 1)]['login'])
    dct = dict(df['song_name'].value_counts())
    print(df.shape)
    print(len(ex_logins))
    df = df[~df.login.isin(ex_logins)]
    print(df.shape)
    return df.reset_index(drop=True)


def make_pivot(df):
    users = pd.unique(df.login)
    sdict = dict()
    songs = list(pd.unique(df.song_name))
    for i in range(len(songs)):
        sdict[songs[i]] = i
    res_df = pd.DataFrame(columns = ['login'] + list(range(len(songs))))
    print(res_df.shape)
    for user in tqdm.tqdm(users):
        tempdf = df.loc[df.login == user]
        urow = [user] + (len(songs)) * [0]
        print(len(urow))
        print(len(songs))
        for index, row in tempdf.iterrows():
            urow[sdict[row['song_name']] + 1] = row['rating']
        res_df.loc[len(res_df)] = urow
    return res_df, sdict


def calc_weighted_rating(v, R, m, C):
    return (v/(v+m) * R) + (m/(m+v) * C)