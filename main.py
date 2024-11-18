import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from collections import Counter
import pickle

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def transform_text(text): #data preprocessing 
	text = text.lower()
	text = nltk.word_tokenize(text)
	y = []
	for i in text:
		if i.isalnum():
			y.append(i)
	text = y[:]
	y.clear()
	for i in text:
		if i not in stopwords.words('english') and i not in string.punctuation:
			y.append(i)
	text = y[:]
	y.clear()
	for i in text:
		y.append(ps.stem(i))
	return " ".join(y)




df_sms = pd.read_csv('sms_spam.csv', encoding='ISO-8859-1')
df_email = pd.read_csv('email_spam.csv', encoding='ISO-8859-1')

#data cleaning

df_sms.drop(columns = ['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace = True)
df_sms.rename(columns = {'v1':'target','v2':'text'}, inplace = True)
df_email.rename(columns = {'Category':'target','Message':'text'}, inplace = True)
encoder = LabelEncoder()
df_sms['target'] = encoder.fit_transform(df_sms['target'])
df_email['target'] = encoder.fit_transform(df_email['target'])
df_sms = df_sms.drop_duplicates(keep = 'first')
df_email = df_email.drop_duplicates(keep = 'first')
df_combined = pd.concat([df_sms, df_email], ignore_index=True)
df_combined = df_combined.drop_duplicates(keep = 'first')
df_combined['num_characters']  = df_combined['text'].apply(len)
df_combined['num_words'] = df_combined['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df_combined['num_sentences'] = df_combined['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

#data preprocessing 

ps = PorterStemmer()
df_combined['transformed_text'] = df_combined['text'].apply(transform_text)

'''
#wordcloud
wc = WordCloud(width=500, height=500, min_font_size=10,background_color='white')
spam_wc = wc.generate(df_combined[df_combined['target'] == 1]['transformed_text'].str.cat(sep = ' '))
plt.figure(figsize = (8,8))
plt.imshow(spam_wc, interpolation = 'bilinear')
plt.axis('off')
plt.show()

wc = WordCloud(width=500, height=500, min_font_size=10,background_color='white')
ham_wc = wc.generate(df_combined[df_combined['target'] == 0]['transformed_text'].str.cat(sep = ' '))
plt.figure(figsize = (8,8))
plt.imshow(ham_wc, interpolation = 'bilinear')
plt.axis('off')
plt.show()'''

spam_corpus = []
for msg in df_combined[df_combined['target'] == 1]['transformed_text'].tolist():
	for word in msg.split():
		spam_corpus.append(word)
ham_corpus = []
for msg in df_combined[df_combined['target'] == 0]['transformed_text'].tolist():
	for word in msg.split():
		ham_corpus.append(word)


#model building

cv = CountVectorizer()
X = cv.fit_transform(df_combined['transformed_text']).toarray()
y = df_combined['target'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 2)
#gnb = GaussianNB()
mnb = MultinomialNB()
#bnb = BernoulliNB()
mnb.fit(X_train,y_train)
y_pred1 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))

#model creation

pickle.dump(cv, open('vectorizer.pkl','wb'))
pickle.dump(mnb, open('model.pkl','wb'))


