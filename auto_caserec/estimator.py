from caserec.evaluation.rating_prediction import RatingPredictionEvaluation
from caserec.evaluation.item_recommendation import ItemRecommendationEvaluation
from caserec.utils.split_database import SplitDatabase
from hyperopt import hp, Trials
from hyperopt import tpe, rand 
from hyperopt import fmin
from hyperopt import STATUS_OK
from .rating_model import RatingModel, RATING_SPACE
from .item_model import ItemModel, ITEM_SPACE
import numpy as np
from statistics import mean, stdev
from .utils import get_fold_paths
import csv
import json
from datetime import date
import time
import os

class AutoEstimator():
    def __init__(self, datapath, predictor='rating', eval_metric='RMSE', eval_rank=10, algo='tpe', max_evals = 20, cross_validate=False, test_percentage=0.2, rank_length=10, sep_read=",", sep_write="\t", n_splits=5, early_stop_split=None, dir_folds = ".", results_path=None, header=1, names=[0,1,2]):
        
        self.predictor = predictor
        self.datapath = datapath
        self.cross_validate = cross_validate
        self.sep_read = sep_read
        self.sep_write = sep_write
        self.header = header
        self.names = names
        self.eval_metric= eval_metric
        self.max_evals = max_evals
        self.best_loss = float('inf')
        self.rank_length = rank_length
        self.eval_rank = eval_rank
        self.optim_results= dict()
        self.best_loss_iter = 0
        self.results_path = results_path
        
        if self.results_path is not None:
            if self.results_path[-1]!="/":
                self.results_path += "/"
                
        else:
            self.results_path=""
        
        field_names = ["Date", "Type", "Iteration", "Metric", "Value", "Configuration", "Time Taken"]
        
        if os.path.exists(self.results_path+"optim_results.csv") is False:
            with open(self.results_path+"optim_results.csv", 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(field_names)           
            
        if self.cross_validate:
            self.n_splits = n_splits
            self.dir_folds = dir_folds
            self.test_percentage=None
        else:
            self.n_splits=1
            self.test_percentage = test_percentage
        
        if early_stop_split is None:
            self.early_stop_split = n_splits
        else:
            self.early_stop_split = early_stop_split-1
            
        if algo=='tpe':
            self.algo= tpe.suggest
        elif algo=='atpe':
            self.algo= atpe.suggest
        else:
            self.algo= rand.suggest
       
        if predictor=='rating':
            self.space = RATING_SPACE
        elif predictor=='item':
            self.space = ITEM_SPACE
            self.eval_metric = self.eval_metric+"@"+str(self.eval_rank)
            
        self.iteration = 0
        self.trials = None
        
        self.make_folds()
    
        
    def make_folds(self):        
        if self.cross_validate:
            SplitDatabase(input_file=self.datapath, 
                          dir_folds = self.dir_folds, 
                          sep_read=self.sep_read, 
                          sep_write=self.sep_write, 
                          n_splits=self.n_splits, 
                          header=self.header, 
                          names=self.names).kfoldcrossvalidation()
        else:
            SplitDatabase(input_file=self.datapath, 
                          dir_folds = self.dir_folds, 
                          sep_read=self.sep_read, 
                          sep_write=self.sep_write,
                          n_splits=self.n_splits, 
                          header=self.header,
                          names=self.names).shuffle_split(test_size=self. test_percentage)
            
        self.train_paths, self.test_paths, self.pred_paths = get_fold_paths(self.dir_folds, self.n_splits)
        
    
        
    def compute(self, config):
        
#         'train_file': self.train_paths[i],
#                  'test_file': self.test_paths[i],
#                  'output_file': self.pred_paths[i],
        train_kwargs = {'sep': self.sep_write,
                        'model': config['type'],
                        'config': config}
        
        if self.predictor=='rating':
            
            results = {'MAE': [], 'RMSE': []}
                   
            for i in range(self.n_splits):
                
                eval_args = {self.pred_paths[i], self.test_paths[i]}
                
                RatingModel(train_file=self.train_paths[i],
                            test_file=self.test_paths[i],
                            output_file=self.pred_paths[i],
                            **train_kwargs)
                
                this_result = RatingPredictionEvaluation(verbose=False).evaluate_with_files(*eval_args)
              
                results['MAE'].append(this_result['MAE'])
                results['RMSE'].append(this_result['RMSE'])
                
                print("Split {} Loss: {}".format(i, results[self.eval_metric][i]))
                
#                 if i>self.early_stop_split:
#                     if mean(results[self.eval_metric]) > self.best_loss:
#                         break;
                        
        else:
            
            train_kwargs['rank_length']=self.rank_length
            results = {self.eval_metric: []}
            
            for i in range(self.n_splits):
                
                eval_args = {self.pred_paths[i], self.test_paths[i]}
                
                ItemModel(train_file=self.train_paths[i],
                          test_file=self.test_paths[i],
                          output_file=self.pred_paths[i],
                          **train_kwargs)
                this_result = ItemRecommendationEvaluation(verbose=False).evaluate_with_files(*eval_args)
                results[self.eval_metric].append(
                    this_result[self.eval_metric]
                )
                
                print("Split {} Loss: {}".format(i, results[self.eval_metric][i]))
                
                if i>self.early_stop_split:
                    if mean(results[self.eval_metric]) > self.best_loss:
                        break;
        return results
                
                
            
    def fit_iter(self, config):
        
        start_time = time.time()
        
        print("Iteration ", self.iteration)
        print("Configuration: ", config)
        
        results = self.compute(config)
        print(results)
        
        
        invert_loss=1
        if self.predictor=='item':
            invert_loss=-1
            
        if self.iteration > 0:
            if invert_loss*mean(results[self.eval_metric])<invert_loss*self.best_loss:
                self.best_loss = mean(results[self.eval_metric])
                self.best_loss_iter = self.iteration
        else:
            self.best_loss = mean(results[self.eval_metric])
            self.best_loss_iter = self.iteration
        
        print("Current Loss: {}".format(mean(results[self.eval_metric])))
        print("Best Loss : {} at iteration {}".format(self.best_loss,self.best_loss_iter))
        
        end_time = time.time()
        
        data_to_write = [str(date.today()), self.predictor, self.iteration, self.eval_metric, mean(results[self.eval_metric]), config, float(end_time-start_time)/60]
        
        with open(self.results_path+"optim_results.csv", 'a+', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(data_to_write) 
                
        if self.predictor=='item':
            for i in range(len(results[self.eval_metric])):
                results[self.eval_metric][i] = 1-results[self.eval_metric][i]
        
   
        self.iteration+=1
        
        return {'loss': mean(results[self.eval_metric]), 'iteration':self.iteration, 'config': config, 'status': STATUS_OK}
    
    def fit(self):
        self.trials = Trials()
        best = fmin(fn=self.fit_iter, space=self.space, algo=self.algo, max_evals=self.max_evals, trials = self.trials)
        return best, self.trials.results
        
        
        
            
        
          
            
        