import requests
import json
import pandas as pd
import tqdm

#   lastfm_get(payload)
#   get_artist_tags(artists, tag_num)


def lastfm_get(payload):
    # define headers and URL
    headers = {'user-agent': 'EDU_PROJECT'}      # здесь впишите что-нибудь другое, если начнут банить аккаунты - 
                                                 # могут всех скопом с таким user-agent'ом подбанить :)
    url = 'https://ws.audioscrobbler.com/2.0/'

    # Add API key and format to the payload
    payload['api_key'] = 'a18850b469ecb6fa86579fb399e96b9b' # подставить свой апи ключ
    payload['format'] = 'json'

    response = requests.get(url, headers=headers, params=payload)
    return response


def get_artist_tags(artists, tag_num):
    df = pd.DataFrame(columns = ['artist', 'tags'])
    for ar in tqdm.tqdm(artists):
        try:
            tags = []
            t1 = lastfm_get({
            'artist' : ar,
            'method': 'artist.getTopTags'})
            data = t1.json()['toptags']['tag']
            ln = min(tag_num, len(data))
            for i in range(ln):
                tags.append(data[i]['name'])
                res = ', '.join(tags)
            df.loc[len(df)] = [ar, res]
        except Exception:
            print('Error occured on string:', ar)
            df.loc[len(df)] = [ar, np.nan]
            continue
    return df

