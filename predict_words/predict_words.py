#!/usr/bin/python

""" 
  In this simple code below, I treined an really simple classifier based on naive bayes algorithm.
  The idea is that you feed your classifier with data and it will distribute each point in a scatterplot and the data 
  will be associated with the label (class/type). After 'feed' your clf with your collected data, you ask for its prediction.
  
  In this specific case, I used the dataset from udacity to treinning my classifier. I am feeding it with words and labels associated 
  with who wrote the words.
  In the end of this feed, my robot will know what are the words associated with each person (and what are the probability 
  of an random word to be from one or another).

"""
    
import sys
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()



clf = GaussianNB()

# in this point I am just feeding my robot with data 
initial_time = time()
clf.fit(features_train, labels_train)
print("Time - in seconds - spend to treining: ", round(time()-initial_time, 3))

# in this point I ask for him to predict who is the owner of the words I am specifying
initial_time = time()
predict_person = clf.predict(features_test)
print("Time - in seconds - spend to predict: ", round(time()-initial_time, 3))

# this is the accuracy of my test - the chance of my training robot be right about the word owner prediction

print(accuracy_score(labels_test, predict_person))