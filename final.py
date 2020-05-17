# -*- coding: utf-8 -*-
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

reviews = load_files('txt_sentoken/')
X,y = reviews.data,reviews.target

with open('X.pickle','wb') as f:
    pickle.dump(X,f)

with open('Y.pickle','wb') as f:
    pickle.dump(y,f)
    
with open('X.pickle','rb') as f:
    X=pickle.load(f)
    
with open('y.pickle','rb') as f:
    y=pickle.load(f)
    
corpus = []
for i in range(0,len(X)):
    review = re.sub(r'/W',' ',str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ',review)
    review = re.sub(r'^[a-z]\s+',' ',review)
    review = re.sub(r'\s',' ',review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test = train_test_split(X,y,test_size=0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)

sent_pred = classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test,sent_pred)

with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)

with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)

with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)

with open('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)
    
sample = ["The rooms are unclean. The Towels are not provided. Servies is not efficient."]
sample = tfidf.transform(sample).toarray()
print(clf.predict(sample))

sample = ["The hotel services are excellent. The staff is very interactive. The food is delicious"]
sample = tfidf.transform(sample).toarray()
print(clf.predict(sample))

if clf.predict(sample)==1:
    import smtplib
    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.starttls()
    MY_ADDRESS = 'abc@gmail.com'
    PASSWORD = '*****'
    s.login(MY_ADDRESS, PASSWORD)
    
    toAddress = "123@gmail.com";
    fromAddress = "abc@gmail.com";
    
    body = """
    Thank You for Your Feedback!!
    """
    s.sendmail(fromAddress, toAddress,  body)
    print(" \n Sent!") 
    s.quit()
    
else:
    import smtplib
    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.starttls()
    MY_ADDRESS = 'abc@gmail.com'
    PASSWORD = '*****'
    s.login(MY_ADDRESS, PASSWORD)
    
    toAddress = "123@gmail.com";
    fromAddress = "abc@gmail.com";
    
    body = """
    Sorry for the Inconvience We will look forward to it.
    """
    s.sendmail(fromAddress, toAddress,  body)
    print(" \n Sent!") 
    s.quit()