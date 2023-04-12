from utils import *

class DataPreparator:
    """
    Creation and preparation of dataset due to minimum count of listening
    df - initial dataset ot interactions
    n - count of listening
    """
    def __init__(self, df, n):      
        rdf = df.groupby(["song_id"])["user_id"].count().reset_index(name="count")
        ids = pd.unique(rdf.loc[rdf['count'] > n]['song_id'])
        res_df = df[df['song_id'].isin(ids)]
        self.df = res_df
        
        self.dataset  = Dataset()
        self.dataset.fit((x for x in self.df['user_id']), (x for x in self.df['song_id']))
        
    def cut_dataset(self):
        # return new dataset with filter
        return self.df
    
    def interaction_weight_matrix(self):
        (interactions, weights) = self.dataset.build_interactions([(x[0], x[1], x[2]) for x in self.df[['user_id', 'song_id', 'rating']].values])  
        return interactions, weights
    
    def user_item_features(self):
        item_ids_range = np.arange(self.dataset.model_dimensions()[1])
        item_ids_predict = np.tile(item_ids_range, self.dataset.model_dimensions()[0]).astype('int32')

        user_ids_range = np.arange(self.dataset.model_dimensions()[0])
        user_ids_predict = np.repeat(user_ids_range, self.dataset.model_dimensions()[1]).astype('int32')
        
        return user_ids_predict, item_ids_predict
    

class Mapping:
    """
    create mapping for user dict and song dict
    """
    def __init__(self, users, songs, df):
        self.users = users
        self.songs = songs
        self.df = df
        
    def prepare_mappings(self):
        usr = {}
        sng = {}
        # song part
        for i in list(pd.unique(self.df['song_id'])):
            sng[i] = self.songs.loc[self.songs['song_id'] == i]['song_name'].item()
        # user part
        for j in list(pd.unique(self.df['user_id'])):
            usr[j] = self.users.loc[self.users['user_id'] == j]['login'].item()
        return usr, sng