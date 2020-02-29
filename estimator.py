from caserec.evaluation.rating_prediction import RatingPredictionEvaluation
from caserec.utils.split_database import SplitDatabase
from hyperopt import hp, Trials
from hyperopt import tpe, atpe, rand 
from hyperopt import fmin
from hyperopt import STATUS_OK
from rating_model import RatingModel, RATING_SPACE
import numpy as np
from statistics import mean, stdev

class AutoEstimator():
    def __init__(self, datapath, predictor='item', eval_metric='RMSE', algo='tpe', max_evals = 20, cross_validate=False, test_percentage=0.2,
                 sep_read=",", sep_write="\t", n_splits=5, dir_folds = ".", header=1, names=[0,1,2]):
        self.predictor = predictor
        self.datapath = datapath
        self.cross_validate = cross_validate
        self.sep_read = sep_read
        self.sep_write = sep_write
        self.header = header
        self.names = names
        self.eval_metric= 'RMSE'
        self.max_evals = max_evals
        self.best_loss = 1000
        
        if self.cross_validate:
            self.n_splits = n_splits
            self.dir_folds = dir_folds
            self.test_percentage=None
        else:
            self.n_splits=1
            self.test_percentage = test_percentage
        
        if algo=='tpe':
            self.algo= tpe.suggest
        elif algo=='atpe':
            self.algo= atpe.suggest
        else:
            self.algo= rand.suggest
       
        if predictor=='item':
            self.space = RATING_SPACE
            
        self.iteration = 0
        self.trials = None
        
        self.make_folds()
    
        
    def make_folds(self):
        
        if self.cross_validate:
            SplitDatabase(input_file=self.datapath, dir_folds = self.dir_folds, sep_read=self.sep_read, sep_write=self.sep_write,
                          n_splits=self.n_splits, header=self.header, names=self.names).kfoldcrossvalidation()
        else:
            SplitDatabase(input_file=self.datapath, dir_folds = self.dir_folds, sep_read=self.sep_read, sep_write=self.sep_write,
                          n_splits=self.n_splits, header=self.header, names=self.names).shuffle_split(test_size=test_percentage)
        
    
    def fit_iter(self, config):
        
        print("Iteration ", self.iteration)
        print("Configuration: ", config)
        results = {'MAE': [], 'RMSE': []}
       
        for i in range(self.n_splits):
            RatingModel(train_file=self.dir_folds+"folds/"+str(i)+"/train.dat", test_file=self.dir_folds+"folds/"+str(i)+"/test.dat",
                                output_file=self.dir_folds+"folds/"+str(i)+"/result.dat", sep=self.sep_write, model=config['type'], config=config)
                       
            fold_results = RatingPredictionEvaluation().evaluate_with_files(self.dir_folds+"folds/"+str(i)+"/result.dat",
                                                                       self.dir_folds+"folds/"+str(i)+"/test.dat")
            results['MAE'].append(fold_results['MAE'])
            results['RMSE'].append(fold_results['RMSE'])
            
            if i>0:
                if mean(results[self.eval_metric]) > self.best_loss:
                    break;
        
        self.iteration+=1
        
        if self.iteration > 0:
            if mean(results[self.eval_metric])<self.best_loss:
                self.best_loss = mean(results[self.eval_metric])
        else:
            self.best_loss = mean(results[self.eval_metric])
            
        return {'loss': mean(results[self.eval_metric]), 'iteration':self.iteration, 'config': config, 'status': STATUS_OK}
    
    def fit(self):
        self.trials = Trials()
        best = fmin(fn=self.fit_iter, space=self.space, algo=self.algo, max_evals=self.max_evals, trials = self.trials, 
                    rstate = np.random.RandomState(50))
        return best, self.trials.results
        
        
        
            
        
          
            
        