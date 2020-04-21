from caserec.recommenders.item_recommendation.bprmf import BprMF
from caserec.recommenders.item_recommendation.group_based_recommender import GroupBasedRecommender
from caserec.recommenders.item_recommendation.itemknn import ItemKNN
from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.recommenders.item_recommendation.paco_recommender import PaCoRecommender  
from caserec.recommenders.item_recommendation.random_rec import RandomRec
from caserec.recommenders.item_recommendation.userknn import UserKNN
from hyperopt import hp

class ItemModel():
    def __init__(self, train_file, test_file, model, config, rank_length=10, output_file=None, sep='\t', output_sep='\t'):
        self.model = model
        self.train_file = train_file
        self.test_file = test_file
        self.sep = sep
        self.config = config
        self.rank_length = rank_length
        if output_file is not None:
            self.output_file = output_file
            self.output_sep = output_sep
        else:
            self.output_file = None
            self.output_sep = None
                
        self.param_options[self.model](self, self.model)
        self.model_options[self.model](self)
        
    def set_knn_params(self, name):
        self.k_neighbors = self.config[name+'_k_neighbors']
        self.similarity_metric = self.config[name+'_similarity_metric']
        self.as_similar_first = self.config[name+'_as_similar_first']
        
    def set_bprmf_params(self, name):
        self.factors = self.config[name+'_factors']
        self.learn_rate = self.config[name+'_learn_rate']
#         self.delta = self.config[name+'_delta']
    
    def set_group_based_params(self, name):
        self.similarity_metric = self.config[name+'_similarity_metric']
        self.k_groups = self.config[name+'_k_groups']
        self.recommender = self.config[name+'_recommender']
        
    def set_most_popular_params(self, name):
        pass
    
    def set_paco_params(self, name):
        self.min_density = self.config[name+'_min_density']
        self.density_low = self.config[name+'_density_low']
        self.n_clusters = self.config[name+'_n_clusters']
        
    def set_rand_params(self, name):
        pass
    
    def get_userknn(self):
        UserKNN(train_file=self.train_file, 
                test_file=self.test_file, 
                output_file=self.output_file, 
                rank_length = self.rank_length,                               
                similarity_metric=self.similarity_metric, 
                k_neighbors=int(self.k_neighbors), 
                sep=self.sep,
                output_sep=self.output_sep).compute(verbose_evaluation=False)
        
    def get_itemknn(self):
        ItemKNN(train_file=self.train_file, 
                test_file=self.test_file,
                output_file=self.output_file, 
                rank_length=self.rank_length,
                similarity_metric=self.similarity_metric, 
                k_neighbors=int(self.k_neighbors), 
                sep=self.sep,
                output_sep=self.output_sep).compute(verbose_evaluation=False)
        
    def get_bprmf(self):
        BprMF(train_file=self.train_file, 
              test_file=self.test_file, 
              output_file=self.output_file, 
              rank_length=self.rank_length,                           
              factors=int(self.factors), 
              learn_rate=self.learn_rate,
              sep=self.sep, 
              output_sep=self.output_sep).compute(verbose_evaluation=False)
        
    def get_group_based(self):
        GroupBasedRecommender(train_files=[self.train_file], 
                              test_file=self.test_file, 
                              output_file=self.output_file, 
                              recommender=self.recommender, 
                              rank_length=self.rank_length, 
                              k_groups=int(self.k_groups), 
                              similarity_metric=self.similarity_metric,
                              sep=self.sep,
                              output_sep=self.output_sep).compute(verbose_evaluation=False)
        
    def get_random(self):
        RandomRec(train_file=self.train_file, 
                  test_file=self.test_file, 
                  output_file=self.output_file, 
                  rank_length=self.rank_length, 
                  sep=self.sep, 
                  output_sep=self.output_sep).compute(verbose_evaluation=False)
        
    def get_paco(self):
        PaCoRecommender(train_file=self.train_file, 
                        test_file=self.test_file, 
                        output_file=self.output_file, 
                        k_row=int(self.n_clusters), 
                        l_col=int(self.n_clusters), 
                        min_density=self.min_density, 
                        density_low=self.density_low).compute(verbose_evaluation=False)
        
    def get_most_popular(self):
        MostPopular(train_file=self.train_file, 
                    test_file=self.test_file, 
                    output_file=self.output_file, 
                    rank_length=self.rank_length,
                    sep=self.sep, 
                    output_sep=self.output_sep).compute(verbose_evaluation=False)
        
    param_options = {
            'userknn': set_knn_params,
            'itemknn': set_knn_params,
            'bprmf': set_bprmf_params,
            'group': set_group_based_params,
            'paco': set_paco_params,
            'random': set_rand_params,
            'most_popular': set_most_popular_params
        }
        
    model_options = {
            'userknn': get_userknn,
            'itemknn': get_itemknn,
            'bprmf': get_bprmf,
            'group': get_group_based,
            'paco': get_paco,
            'random': get_random,
            'most_popular': get_most_popular         
        }
        
        
ITEM_SPACE = hp.choice('recommender_type',[
    {
        'type': 'itemknn',
        'itemknn_k_neighbors': hp.quniform('itemknn_k_neighbors', 1, 100, 1),
        'itemknn_similarity_metric': hp.choice('itemknn_similarity_metric', ['cosine']),#, 'jaccard', 'euclidean', 'correlation', 'hamming']),
        'itemknn_as_similar_first': hp.choice('itemknn_as_similar_first', [True, False])        
    },
    
    {
        'type': 'bprmf',
        'bprmf_factors': hp.quniform('bprmf_factors', 10, 200, 1),
        'bprmf_learn_rate': hp.uniform('bprmf_learn_rate', 0.001, 0.1),
    },
    
    {
        'type': 'most_popular'
    },
    
    {
        'type': 'random',    
    },
    
#     {
#         'type': 'group',
#         'group_similarity_metric': hp.choice('group_similarity_metric', ['cosine']),#, 'jaccard', 'euclidean', 'correlation', 'hamming'])
#         'group_k_groups': hp.quniform('group_k_groups', 2, 30, 1),
#         'group_recommender': hp.choice('group_recommender', ['UserKNN', 'ItemKNN', 'MostPopular', 'BPRMF'])
        
#     },
    
#     {
#         'type': 'paco',
#         'paco_min_density': hp.quniform('paco_min_density', 0.3, 0.9, 0.1),
#         'paco_density_low': hp.uniform('paco_density_low', 0.001, 0.01),
#         'paco_n_clusters': hp.quniform('paco_n_clusters', 2, 20, 1)
    
#     },
    
    {
        'type': 'userknn',
        'userknn_k_neighbors': hp.quniform('userknn_k_neighbors', 1, 100, 1),
        'userknn_similarity_metric': hp.choice('userknn_similarity_metric', ['cosine']),#, 'jaccard', 'euclidean', 'correlation', 'hamming']),
        'userknn_as_similar_first': hp.choice('userknn_as_similar_first', [True, False])  
    }    
])