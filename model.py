from utils import *

class Model:
    """
    """
    def __init__(self):
        pass
    
    def train_test_split(self, interactions, weights):
        self.train, self.test = random_train_test_split(interactions, test_percentage=0.2, random_state = 42)
        self.train_weights, self.test_weights = random_train_test_split(weights, test_percentage=0.2, random_state = 42)
        
    def fit(self, epochs=65, **params):
        self.model = LightFM(**params).fit(self.train_weights, epochs=epochs)
        
    def metrics(self):
        test_precision_at_5 = precision_at_k(self.model, test_interactions = self.test_weights, train_interactions = self.train_weights, k=5, check_intersections=False).mean()
        test_recall_at_5 = recall_at_k(self.model, test_interactions = self.test_weights, train_interactions = self.train_weights, k=5, check_intersections=False).mean()

        print('Precision: test %.2f' % (test_precision_at_5))
        print("Recall: test %.2f" % (test_recall_at_5))

    def save(self, filename):
        pickle.dump(self.model, open(filename, 'wb'))    
        
    def make_recommendation(self, n, ndf, umap, smap, ufeat, sfeat, user_id=None): 
        """
        n - num of recommendation
        ndf - initial interaction dataset
        sdf - song mapping
        """
        if user_id is None or user_id not in umap.keys(): ## checking user - existing id or new
            ## new user
            """
            Calculation of universal recommendation for our dataset
            """
            return get_n_popular_recommendations(ndf, smap, n=n)
        else:
            ## existing user
            """
            Prediction for existing user
            """
            user_index = {k: v for v, k in enumerate(list(umap.keys()))}
            song_index = dict(enumerate(list(smap.keys())))
            shape = len(smap.keys())
            u_ind = user_index[user_id]
            prediction = self.model.predict(user_ids = ufeat[shape * u_ind : shape * (u_ind + 1)], item_ids = sfeat[shape * u_ind : shape * (u_ind + 1)])
            sorted_pred = pd.Series(prediction).nlargest(len(prediction)) # sorted predictions by score
            
            already_heard = get_what_user_heard(user_id, ndf, smap)
            result = []
            for i in sorted_pred.index.values.tolist():
                if len(result) < n:
                    try:
                        song = smap[song_index[i]]
                        if song not in already_heard:
                            result.append(song)
                    except Exception:
                        continue
            return result
