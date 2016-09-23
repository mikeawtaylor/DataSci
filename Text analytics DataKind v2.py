# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 10:34:23 2016

1.	M/D
2.	Phone
3.	First Set
4.	First Actual
5.	Second Set
6.	Second Actual

@author: mike.a.taylor
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.externals import joblib

##############################################################################
# Constants & Dummy Data
##############################################################################

# Constants - toggle these as required
vecLen = 1000
ngramLen = 3
testSize = 0.33
hashing = False

## dummy corpus list - "working... convert to real data when available"
#from sklearn.datasets import fetch_20newsgroups
#categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
#twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
#twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
#corpus = twenty_train.data
#corpLabel = twenty_train.target

##############################################################################
# Functions
##############################################################################

def tokenizeEtc(corpus):
    # todo - ignore the descriptions which are empty    
    
    # create ngrams
    ngramVectorizer = CountVectorizer(ngram_range=(1, ngramLen),token_pattern=r'\b\w+\b', min_df=1)
    ngrammer = ngramVectorizer.build_analyzer()
    
    # convert to hashable
    # bit hacky, better to do it as a pipeline in sklearn pipeline... 
    #http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html "Building a Pipeline"
    corpusTmp = []
    for i in range(0,len(corpus)):
        try:
            corpusTmp.append(ngrammer(corpus[i]))
        except:
            corpusTmp.append("house") # used a known "duff word" as a excet word
    corpusFinal = []
    corpusBody = ""
    for i in range(0,len(corpusTmp)):
        for j in range(0,len(corpusTmp[i])):
            corpusBody = corpusBody + " " + corpusTmp[i][j]
        corpusFinal.append(corpusBody)
        corpusBody = ""
    
    # hashing the post ngrammed (slightly horrible) text bodies
    hv = HashingVectorizer(n_features=vecLen)
    hashedcorpus = hv.transform(corpusFinal)
    
    # tfidf the array of ngrams (or leave as regular, if hashing is not chosen)
    transformer = TfidfTransformer(norm = None, smooth_idf = True, sublinear_tf=False)
    if hashing == True:
        tfIdfVecLtmp = transformer.fit_transform(hashedcorpus) 
    else:
        vectorizer = TfidfVectorizer(min_df=1)
        tfIdfVecLtmp = vectorizer.fit_transform(corpusFinal)
    tfIdfVecL = tfIdfVecLtmp.toarray()
    
    # bit hacky, cleans up negative numbers (adds 3) shouldn't be necessary, 
    #something going slightly wrong here, suspect it's the sklearn imp.
    for i in range(0,len(tfIdfVecL)):
        for j in range(0,len(tfIdfVecL[i])):
            if tfIdfVecL[i][j] < 0: 
                tfIdfVecL[i][j] = 0
    
    print("Data Preparation Complete")
    return(tfIdfVecL)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

##############################################################################
# Main Code
##############################################################################

print("DataKind Text Analytics Beta")
print("--------------------------------------------------")
print("")
print("Hashing - ", hashing)
print("Hash Vector Length - ", vecLen)
print("Ngram length - ", ngramLen)
print("Test Split - ", testSize)

# Import the Data
textDataFrameFull = pd.read_csv(r"UKLandAndFarms_full_labelledv2.csv", encoding = "ISO-8859-1")
corpusFull = np.asarray(textDataFrameFull['Text Description'])
textDataFrameLabelled = textDataFrameFull[(textDataFrameFull['keep']==1) | (textDataFrameFull['keep']==0)]
textDataFrameLabelled.reset_index()
corpLabel = np.asarray(textDataFrameLabelled['keep'])
corpusSmall = np.asarray(textDataFrameLabelled['Text Description'])

goodList = ["productive", "arable", "pasture", "farm", "agricultural", "rural farming", "pastureland", "livestock"]
#badList =  ["house", "holiday letting", "farmhouse", "residential", "cottage", "bedroom", "bed", "home"]

# todo - add some stemming etc. here to remove punctuation, lower case it, remove plurals etc. etc.

tfIdfVecL = tokenizeEtc(corpusSmall)

# Split data for  test and training
X_train, X_test, y_train, y_test = train_test_split(tfIdfVecL, corpLabel, test_size=testSize, random_state=42)

## Multinomial Naive Bayes
## Train
MNBclf = MultinomialNB().fit(X_train, y_train)
print ("Multinomial Naive Bayes Classification Training Complete")
##Test
predicted = MNBclf.predict(X_test)
#print ("Multinomial Naive Bayes Classification Accuracy -", round(100*np.mean(predicted == y_test),1), "%")

# Decision Tree
# Train
DTclf = DecisionTreeClassifier(max_depth = 5).fit(X_train, y_train)
print ("Multinomial Naive Bayes Classification Training Complete")
# Test 
predicted = DTclf.predict(X_test)
print ("Decision Tree Classification Accuracy -", round(100*np.mean(predicted == y_test),1), "%")

# PLot confusion matrices
cm = confusion_matrix(y_test, predicted)
print (cm)
#plot_confusion_matrix(cm, title = "Not Normalised")
#plt.show()
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix(cm_normalized, title = "Normalised")
plt.show()

# Save the model
joblib.dump(DTclf, 'DT_model.pkl')
print ("ML Model Saved as .pkl")
DTclf = joblib.load('DT_model.pkl') 

# todo - add in the outputs, marking up the csv/excel with the yes/no info
# todo - need to clean up the data tokenizer, something in the data set appears to be breaking it
tfIdfVecLFull = tokenizeEtc(corpusFull)
predicted = DTclf.predict(tfIdfVecLFull)
#print(predicted)


