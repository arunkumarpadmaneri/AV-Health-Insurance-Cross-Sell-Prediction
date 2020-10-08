import joblib
import pandas as pd 
from sklearn import metrics
from sklearn import tree

#param: fold - validation set
def run(fold):
	# read the training data with folds
	df = pd.read_csv("../input/train_folds.csv")
	
	#get the training data(removing validation set)
	df_train = df[df.kfold != fold].reset_index(drop=True)

	# get validation set
	df_validation = df[df.kfold == fold].reset_index(drop=True)

	x_train = df_train.drop('label', axis=1).values
	y_train = df_train.label.values

	x_valid = df_validation.drop('label', axis=1).values
	y_valid = df_validation.label.values

	clf = tree.DecisionTreeClassifier()
	clf.fit(x_train,y_train)
	preds = clf.predict(x_valid)

	#calculate & print accuracy
	accuracy = metrics.accuracy_score(y_valid, preds)
	print(f"Fold={fold},Accuracy={accuracy}")

	#save the model
	joblib.dump(clf,f"../models/dt_{fold}.bin")

if __name__ == "__main__":
	run(fold=0)
	run(fold=1)
	run(fold=2)
	run(fold=3)
	run(fold=4)
	run(fold=5)

