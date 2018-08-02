# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

def metrics(y_test, y_pred):
	# Making the Confusion Matrix
	cm = confusion_matrix(y_test, y_pred)
	print(cm)

	# Printing different metric scores
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)
	print('precision: ', precision, '\nrecall: ', recall, '\nf1: ', f1)

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [1]].values
y = dataset.iloc[:, [2,3,4,5,6,7]].max(axis=1)
y = y.values

# Splitting the dataset into the Training set and Test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# transform matrix of plots into lists to pass to a TfidfVectorizer
train_X = [x[0].strip() for x in X_train.tolist()]
test_X = [x[0].strip() for x in X_test.tolist()]

# Create pipeline for the classifier
classifier = Pipeline([('tfidf', TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
        										lowercase=True, max_df=0.25, max_features=None, min_df=10,
        										ngram_range=(1, 2), norm='l2', preprocessor=None, smooth_idf=True,
        										stop_words='english', strip_accents=None, sublinear_tf=False, use_idf=True)),
		             							('classifier', OneVsRestClassifier(LinearSVC(), n_jobs=1))])
# Fit the classifier to the training set
classifier.fit(train_X, y_train)

# Test the classifier accuracy and print the results
y_pred = classifier.predict(test_X)
metrics(y_test, y_pred)

# Save the model for production
joblib.dump(classifier, 'classifier_light.pkl')