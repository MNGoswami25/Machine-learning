import pandas as pd   # library used for working with data sets. It has functions for analyzing, cleaning, exploring, and manipulating data.
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split  #  to split a dataset into two (or more) subsets: one for training the machine learning model and the other for testing its performance.
from sklearn.feature_extraction.text import CountVectorizer

#loading
data = pd.read_csv(r'D:\Machine-learning\Spam.csv')

#print(data.info())

# split data into x and y
X = data['EmailText'].values
y = data['Label'].values

#spit data into trainning and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

cv = CountVectorizer()  #  CountVectorizer class is used to convert a collection of text documents into a matrix of token counts. It is a popular technique used to preprocess text data for machine learning tasks.
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

classifier = SVC(kernel='rbf',random_state = 10)
classifier.fit(X_train, y_train)

print(classifier.score(X_test,y_test))