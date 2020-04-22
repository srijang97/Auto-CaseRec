from caserec.recommenders.rating_prediction.itemknn import ItemKNN
from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
from caserec.recommenders.rating_prediction.most_popular import MostPopular
from caserec.recommenders.rating_prediction.random_rec import RandomRec
from caserec.recommenders.rating_prediction.svd import SVD
from caserec.recommenders.rating_prediction.svdplusplus import SVDPlusPlus
from caserec.recommenders.rating_prediction.userknn import UserKNN
from hyperopt import hp

class RatingModel():
    def __init__(self, train_file, test_file, model, config, output_file=None, sep='\t', output_sep='\t'):
        self.model = model
        self.train_file = train_file
        self.test_file = test_file
        self.sep = sep
        self.config = config
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
        
    def set_mf_params(self, name):
        self.factors = self.config[name+'_factors']
        self.learn_rate = self.config[name+'_learn_rate']
        self.epochs = self.config[name+'_epochs']
        self.delta = self.config[name+'_delta']
    
    def set_most_popular_params(self, name):
        pass
    
    def set_svd_params(self, name):
        self.factors = self.config['factors']
        
    def set_rand_params(self, name):
        self.uniform = self.config['uniform']    
    
    def get_userknn(self):
        UserKNN(train_file=self.train_file, test_file=self.test_file, output_file=self.output_file,
                               similarity_metric=self.similarity_metric, k_neighbors=int(self.k_neighbors), sep=self.sep,
                               output_sep=self.output_sep).compute(verbose_evaluation=False)
    def get_itemknn(self):
        ItemKNN(train_file=self.train_file, test_file=self.test_file, output_file=self.output_file,
                               similarity_metric=self.similarity_metric, k_neighbors=int(self.k_neighbors), sep=self.sep,
                               output_sep=self.output_sep).compute(verbose_evaluation=False)
    def get_svdplusplus(self):
        SVDPlusPlus(train_file=self.train_file, test_file=self.test_file, output_file=self.output_file,
                                  factors=int(self.factors), learn_rate=self.learn_rate, epochs=int(self.epochs), delta=self.delta,
                                  sep=self.sep, output_sep=self.output_sep).compute(verbose_evaluation=False)
    def get_mf(self):
        MatrixFactorization(train_file=self.train_file, test_file=self.test_file, output_file=self.output_file,
                                  factors=int(self.factors), learn_rate=self.learn_rate, epochs=int(self.epochs), delta=self.delta,
                                  sep=self.sep, output_sep=self.output_sep).compute(verbose_evaluation=False)
    def get_random(self):
        RandomRec(train_file=self.train_file, test_file=self.test_file, output_file=self.output_file, 
                                 uniform=self.uniform, sep=self.sep, output_sep=self.output_sep).compute(verbose_evaluation=False)
    def get_svd(self):
        SVD(train_file=self.train_file, test_file=self.test_file, output_file=self.output_file,
                           factors=int(self.factors), sep=self.sep, output_sep=self.output_sep).compute(verbose_evaluation=False)
    def get_most_popular(self):
        MostPopular(train_file=self.train_file, test_file=self.test_file, output_file=self.output_file, 
                           sep=self.sep, output_sep=self.output_sep).compute(verbose_evaluation=False)
        
    param_options = {
            'userknn': set_knn_params,
            'itemknn': set_knn_params,
            'svdplusplus': set_mf_params,
            'mf': set_mf_params,
            'svd': set_svd_params,
            'random': set_rand_params,
            'most_popular': set_most_popular_params
        }
        
    model_options = {
            'userknn': get_userknn,
            'itemknn': get_itemknn,
            'svdplusplus': get_svdplusplus,
            'mf': get_mf,
            'svd': get_svd,
            'random': get_random,
            'most_popular': get_most_popular
        
        }
        
        
RATING_SPACE = hp.choice('recommender_type',[
    {
        'type': 'itemknn',
        'itemknn_k_neighbors': hp.quniform('itemknn_k_neighbors', 1, 100, 1),
        'itemknn_similarity_metric': hp.choice('itemknn_similarity_metric', ['cosine', 'jaccard', 'euclidean', 'correlation', 'hamming']),
        'itemknn_as_similar_first': hp.choice('itemknn_as_similar_first', [True, False])        
    },
    
    {
        'type': 'mf',
        'mf_factors': hp.quniform('mf_factors', 10, 200, 1),
        'mf_learn_rate': hp.uniform('mf_learn_rate', 0.001, 0.1),
        'mf_epochs': hp.quniform('mf_epochs', 10, 30, 1),
        'mf_delta': hp.uniform('mf_delta', 0.001, 0.1)
    },
    
    {
        'type': 'most_popular'
    },
    
    {
        'type': 'random',
        'uniform': hp.choice('uniform', [True, False])        
    },
    
    {
        'type': 'svd',
        'factors': hp.quniform('factors', 10, 200, 1)
    },
    
#     {
#         'type': 'svdplusplus',
#         'svdplusplus_factors': hp.quniform('svdplusplus_factors', 10, 200, 1),
#         'svdplusplus_learn_rate': hp.uniform('svdplusplus_learn_rate', 0.001, 0.1),
#         'svdplusplus_epochs': hp.quniform('svdplusplus_epochs', 10, 30, 1),
#         'svdplusplus_delta': hp. uniform('svdplusplus_delta', 0.001, 0.1)        
#     },
    
    {
        'type': 'userknn',
        'userknn_k_neighbors': hp.quniform('userknn_k_neighbors', 1, 100, 1),
        'userknn_similarity_metric': hp.choice('userknn_similarity_metric', ['cosine', 'jaccard', 'euclidean', 'correlation', 'hamming']),
        'userknn_as_similar_first': hp.choice('userknn_as_similar_first', [True, False])  
    }    
])