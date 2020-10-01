#Importing Libraries
import numpy as np
import pandas as pd

#importing Dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter="\t")


#Cleaning of dataset

import re
#Selecting the useful things
review = re.sub('[^a-zA-z]',' ',dataset['Review'][0])
#Changing case of alpahbet
review = review.lower()
#Splitting the sentence
review = review.split()


#Importing NLP library
import nltk
nltk.download('stopwords')
#Removing unuseful words( like - is,the,this)
from nltk.corpus import stopwords # to use stopword in spyder notebook
review = [word for word in review if not word in stopwords.words('english')]

#stemming(Finding root word from word by removing root word)
from nltk.stem.porter import PorterStemmer
# can also use Leminizer in place of PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review]

review=" ".join(review)
#For whole dataset
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in stopwords.words('english')]
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    review=" ".join(review)  
    corpus.append(review)

#Making Features and labels 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
features = cv.fit_transform(corpus).toarray() 
labels = dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.25,random_state=0)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,p=2, metric='minkowski' )
classifier.fit(features_train,labels_train)

#Predicting the class labels 
labels_pred =classifier.predict(features_test)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pred)

#Accuracy Score
from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pred)








