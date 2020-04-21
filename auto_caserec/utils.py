def get_fold_paths(dir_folds, n_splits):
    train_path = []
    test_path = []
    pred_path = []
    
    for i in range(n_splits):
        train_path.append(dir_folds+"folds/"+str(i)+"/train.dat")
        test_path.append(dir_folds+"folds/"+str(i)+"/test.dat")
        pred_path.append(dir_folds+"folds/"+str(i)+"/result.dat")
        
    return train_path, test_path, pred_path