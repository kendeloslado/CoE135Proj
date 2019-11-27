import os
import numpy as np
import re
import csv
import time
from sklearn.naive_bayes import MultinomialNB

# Creates dictionary from all the emails in the directory
def build_dictionary(dir):
  # Read the file names
  emails = os.listdir(dir)
  emails.sort()
  # Array to hold all the words in the emails
  dictionary = []

  # Collecting all words from those emails
  for email in emails:
    m = open(os.path.join(dir, email))
    for i, line in enumerate(m):
      if i == 2: # Body of email is only 3rd line of text file
        words = line.split()
        dictionary += words

  # We now have the array of words, whoch may have duplicate entries
  dictionary = list(set(dictionary)) # Removes duplicates

  # Removes puctuations and non alphabets
  for index, word in enumerate(dictionary):
    if (word.isalpha() == False) or (len(word) == 1):
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

  return labels_matrix 

start_time = time.time()
train_dir = '/Users/Ken Delos Lado/OneDrive/Documents/UPD/CoE_135/lingspam_public.tar/lingspam_public/lingspam_public/train_data'
dictionary = build_dictionary(train_dir)
features_train = build_features(train_dir, dictionary)
labels_train = build_labels(train_dir)

classifier = MultinomialNB()
classifier.fit(features_train, labels_train)

test_dir = '/Users/Ken Delos Lado/OneDrive/Documents/UPD/CoE_135/lingspam_public.tar/lingspam_public/lingspam_public/test_data'
features_test = build_features(test_dir, dictionary)
labels_test = build_labels(test_dir)

accuracy = classifier.score(features_test, labels_test)
print("The accuracy is %s" % accuracy)
# print(dictionary)
with open('dictionary.csv', "w") as csv_file:
  writer = csv.writer(csv_file, delimiter=' ')
  for line in dictionary:
      writer.writerow(line)
print("---Program was executed in %s seconds ---" % (time.time() - start_time))
"""
This code in particular is extracted directly from 
https://github.com/alameenkhader/spam_classifier?files=1

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