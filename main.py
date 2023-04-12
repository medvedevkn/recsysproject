from model import *
from data_preparation import *
import utils
import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv('data/pivot_df.csv')
    users = pd.read_csv('data/u_index.csv')
    songs = pd.read_csv('data/s_index.csv')

    dp = DataPreparator(df, 30)
    user_feat, song_feat = dp.user_item_features()
    interactions, weights = dp.interaction_weight_matrix()
    df = dp.cut_dataset()
    
    umap, smap = Mapping(users, songs, df).prepare_mappings()

    params = {'loss':'warp', 'no_components':199, 'learning_rate':0.01, 'max_sampled':20, 'random_state':42}

    mym = Model()
    mym.train_test_split(interactions, weights)
    mym.fit(65, **params)
    mym.metrics()
    print(mym.make_recommendation(7, df, umap, smap, user_feat, song_feat, user_id=None))