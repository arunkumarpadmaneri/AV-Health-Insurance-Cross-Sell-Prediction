# load the libraries
import numpy as np 
import pandas as pd 

from sklearn import datasets
from sklearn.model_selection import StratifiedKFold,KFold


# if target is imbalanced data we use startified kfold validation 
def create_folds_stratified(data,target):
	#create a new column kfold fill with it -1
	data["kfold"] = -1
	# randomize the rows of data
	data =  data.sample(frac=1).reset_index(drop=True)
	y = data[target].values
	# initiate the kfold class from model_selection module
	kf =  StratifiedKFold(n_splits=5)
	# kf = model_selection.KFold()
	for f, (t_,v_) in enumerate(kf.split(X = data, y = y)):
		data.loc[v_,'kfold']  = f

	#save the new csv with kfold column
	data.to_csv("../input/train_folds.csv",index=False)

# if target is imbalanced data we use startified kfold validation 
def create_folds(data,target):
	#create a new column kfold fill with it -1
	data["kfold"] = -1
	# randomize the rows of data
	data =  data.sample(frac=1).reset_index(drop=True)
	y = data[target].values
	# initiate the kfold class from model_selection module
	kf =  KFold(n_splits=5)

	for fold, (train_,validation_) in enumerate(kf.split(X = data)):
		data.loc[validation_,'kfold']  = fold

	#save the new csv with kfold column
	data.to_csv("../input/train_folds.csv",index=False)

#TODO:need to implement group kfold, group kfold with stratified kfold

if __name__=="__main__":
	data = pd.read_csv("../input/train.csv")
	create_folds(data,"Response")
	foldcsv = pd.read_csv("../input/train_folds.csv")
	print(foldcsv.tail())