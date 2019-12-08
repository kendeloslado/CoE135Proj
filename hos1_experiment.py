import os
import csv
import nltk
import time
#nltk.download('stopwords') #run this if you don't have nltk downloaded yet
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords 

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:  #Body of email is only 3rd line of text file
                    words = line.split()
                    words = list(dict.fromkeys(words))
                    all_words += words
    dictionary = Counter(all_words)

    stop_words = set(stopwords.words('english'))
    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
        elif item in stop_words:
            del dictionary[item]
    dictionary = dictionary.most_common()
    return dictionary

"""def extract_features(mail_dir):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), len(dictionary)))
    docID = 0
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1
    return features_matrix"""

def prob_given_ham(test_dir):
    emails = [os.path.join(test_dir,f) for f in os.listdir(test_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:  #Body of email is only 3rd line of text file
                    words = line.split()
                    words = list(dict.fromkeys(words))
                    all_words += words
    dictionary = all_words

    stop_words = set(stopwords.words('english'))
    #list_to_remove = dictionary.keys()
    for item in list(dictionary):
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
        elif item in stop_words:
            del dictionary[item]
    dictionary = dictionary.most_common()
    ham = dictionary
    return ham
##########################################################

start_time = time.time()

##########################################################
# Create a dictionaryham of words with its frequency
 
train_dir = '/Users/User/Desktop/CoE 135 Project/2/train_data/train_ham'
dictionary = make_Dictionary(train_dir)
dictionaryham = dictionary
print(dictionaryham[1][7]) 
# Prepare feature vectors per training mail and its labels

"""size_ham = 0
with open('dictionaryham.csv', "w", newline='') as csv_file:
  writer = csv.writer(csv_file, delimiter=',')
  for line in dictionary:
      writer.writerow(line)
      size_ham = size_ham + 1
print("number of ham words: %i" % (size_ham))"""

############################################################

# Create a dictionaryspam of words with its frequency
 
train_dir = '/Users/User/Desktop/CoE 135 Project/2/train_data/train_spam'
dictionary = make_Dictionary(train_dir)
dictionaryspam = dictionary 
#print(dictionaryspam)
# Prepare feature vectors per training mail and its labels
 


"""size_spam = 0
with open('dictionaryspam.csv', "w", newline='') as csv_file:
  writer = csv.writer(csv_file, delimiter=',')
  for line in dictionary:
      writer.writerow(line)
      size_spam = size_spam + 1
print("number of spam words: %i" % (size_spam))"""
##############################################################
# TEST #
#test_dir = '/Users/User/Desktop/CoE 135 Project/2/test_data'
#ham = prob_given_ham(test_dir)
#print(ham)


print("---Program was executed in %s seconds ---" % (time.time() - start_time))