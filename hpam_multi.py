import os
import numpy as np
import re
import csv
import time
import threading
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from collections import Counter
from nltk.corpus import stopwords
#from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Process
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
# Creates dictionary from all the emails in the directory
"""
def build_dictionary(dir):
  # Read the file names
  # emails = os.listdir(dir)
  # emails.sort()
  # Array to hold all the words in the emails
  mails = [os.path.join(dir, f) for f in os.listdir(dir)]
  all_words = []
  # Collecting all words from those emails
  for email in mails:
      with open(email) as m:
        for i,line in enumerate(m):
          if i == 2: # Body of email is only 3rd line of text file
            words = line.split()
            all_words += words
  dictionary = Counter(all_words)
  # We now have the array of words, which may have duplicate entries
  #filter_words = list(set(dictionary)) # Removes duplicates
  filter_words = dictionary.keys()
  # dictionary = list(set(dictionary))
  # Removes puctuations and non alphabets
  stop_words = set(stopwords.words('english'))
  for index in list(filter_words):
    if (index.isalpha() == False) or (len(index) == 1) or (index in stop_words):
      del dictionary[index]
  dictionary = dictionary.most_common()
  return dictionary
"""

def build_dictionary(dir):
  # Read the file names  
  pool = ThreadPool()
  #emails = pool.map(os.listdir, dir)
  emails = os.listdir(dir)
  #pool.close()
  #pool.join()
  emails.sort()
  # Array to hold all the words in the emails
  dictionary = []

  m = pool.map(open(os.path.join(dir,email)), emails)
  # Collecting all words from those emails
  #for email in emails:
  #  m = open(os.path.join(dir, email))
  for i, line in enumerate(m):
      if i == 2: # Body of email is only 3rd line of text file
        words = line.split()
        #dictionary.append(words)
        dictionary += words
  # We now have the array of words, whoch may have duplicate entries
  dictionary = list(set(dictionary)) # Removes duplicates
  stop_words = set(stopwords.words('english'))
  # Removes puctuations and non alphabets
  for index, word in enumerate(dictionary):
    if (word.isalpha() == False) or (len(word) == 1) or (word in stop_words):
      del dictionary[index]

  return dictionary


def build_features(dir, dictionary):
  # Read the file names
  emails = os.listdir(dir)
  emails.sort()
  # ndarray to have the features
  features_matrix = np.zeros((len(emails), len(dictionary)))

  # collecting the number of occurances of each of the words in the emails
  for email_index, email in enumerate(emails):
    m = open(os.path.join(dir, email))
    for line_index, line in enumerate(m):
      if line_index == 2:
        words = line.split()
        for word_index, word in enumerate(dictionary):
          features_matrix[email_index, word_index] = words.count(word)

  return features_matrix

def build_labels(dir):
  # Read the file names  
  emails = os.listdir(dir)
  emails.sort()
  # ndarray of labels
  labels_matrix = np.zeros(len(emails))

  for index, email in enumerate(emails):
    labels_matrix[index] = 1 if re.search('spms*', email) else 0
  #print(len(emails) - np.count_nonzero(labels_matrix))
  #print(np.count_nonzero(labels_matrix))

  return labels_matrix 

"""dictionary = threading.Thread(target=build_dictionary, args=(train_dir))
features_train = threading.Thread(target=build_features,args=(train_dir))
labels_train = threading.Thread(target=build_labels, args=(dir))
#
dictionary.start()
features_train.start()
labels_train.start()

dictionary = build_dictionary(train_dir)
features_train = build_features(train_dir, dictionary)
labels_train = build_labels(train_dir)

dictionary.join()
features_train.join()
labels_train.join()
"""
if __name__ == '__main__':
    start_time = time.time() #start timing the code
    train_dir = '/Users/Ken Delos Lado/OneDrive/Documents/UPD/CoE_135/lingspam_public.tar/lingspam_public/lingspam_public/train_data'
    #with Pool(3) as dictionary:
    dictionary = Pool(processes=2)
    dictionary = build_dictionary(train_dir)
    #dictionary = build_dictionary(train_dir)  
    #with Pool(3) as features_train:
    features_train = build_features(train_dir, dictionary)
    #with Pool(3) as labels_train:
    labels_train = build_labels(train_dir)
      
    print("Number of train ham mail is %i" % (len(labels_train) - np.count_nonzero(labels_train)))
    print("Number of train spam mail is %i" % (np.count_nonzero(labels_train)))
    classifier = MultinomialNB()
    classifier.fit(features_train, labels_train)

    test_dir = '/Users/Ken Delos Lado/OneDrive/Documents/UPD/CoE_135/lingspam_public.tar/lingspam_public/lingspam_public/test_data'
    #with Pool(2) as features_test:
    features_test = build_features(test_dir, dictionary)
    #with Pool(2) as labels_test:
    labels_test = build_labels(test_dir)
    print("Number of test ham mail is %i" % (len(labels_test) - np.count_nonzero(labels_test)))
    print("Number of test spam mail is %i" % (np.count_nonzero(labels_test)))
    accuracy = classifier.score(features_test, labels_test)
    #with Pool(2) as predict_feat_test:
    predict_feat_test = classifier.predict_proba(features_test)
    print("The accuracy is %s" % accuracy)
        # print(features_test)
        #print(predict_feat_test)
        #print(labels_train)
        #print(predict_label)
        #print(dictionary)
    with open('dictionary.csv', "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=' ')
    #writer = csv.writer(csv_file, delimiter=',')
        for line in dictionary:
            writer.writerow(line)
    with open('stats.csv', "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["P_Ham", "P_Spam"])
        writer.writerows(predict_feat_test)
    with open('labels.csv', "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(labels_test)
    with open('features.csv', "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(features_test)
    with open('labelstrain.csv', "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(labels_train)
    with open('featurestrain.csv', "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(features_train)

    #model1 = LinearSVC()
    #model1.fit()
    print("---Program was executed in %s seconds ---" % (time.time() - start_time))
            

"""
This code in particular is extracted directly from 
https://github.com/alameenkhader/spam_classifier?files=1
Credits to Alameen Khader for this code

References used
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
https://stackoverflow.com/questions/50515740/naivebayes-multinomialnb-scikit-learn-sklearn
https://towardsdatascience.com/spam-or-ham-introduction-to-natural-language-processing-part-2-a0093185aebd
http://www.insightsbot.com/bag-of-words-algorithm-in-python-introduction/
https://www.geeksforgeeks.org/bag-of-words-bow-model-in-nlp/
https://www.freecodecamp.org/news/an-introduction-to-bag-of-words-and-how-to-code-it-in-python-for-nlp-282e87a9da04/
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
https://www.programcreek.com/python/example/85919/sklearn.naive_bayes.MultinomialNB
https://numpy.org/devdocs/user/quickstart.html
https://docs.python.org/3/library/re.html
https://appliedmachinelearning.blog/2017/01/23/email-spam-filter-python-scikit-learn/
https://www.dropbox.com/s/yjiplngoa430rid/ling-spam.zip
https://scikit-learn.org/stable/modules/svm.html
https://scikit-learn.org/stable/modules/classes.html
https://docs.python.org/3/library/os.html#os.listdir
https://docs.python.org/3/library/os.html#os.name
https://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution
"""