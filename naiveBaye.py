import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Language Detection.csv")
data.head(10)

#Separate Independent and Dependent features
x = data["Text"]
y = data["Language"]

#create labelEncoder
le = LabelEncoder()
#convert string labels into numbers
#Important process for it to be machine readable
label_encoded = le.fit_transform(y)

dataList = []
#Clean the data
for i in x:
    i = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', i)
    i = re.sub(r'[[]]', ' ', i)
    i = i.lower()
    dataList.append(i)


#Bag of Words (Histogram)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(dataList).toarray()

#performing the test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size =0.2)
print(x_test.shape)
print(x_train.shape)
print(y_test.shape)
print(y_train.shape)

#Trained model using training set
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB
model.fit(x_train, y_train)

#Language Predicition
y_predict = model.predict(x_test)






