import streamlit as st
import pickle
import string 
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


ps = PorterStemmer()

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

cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_msg = st.text_area("Enter the message")

if st.button('Predict'):

	#text preprocessing 

	transformed_msg = transform_text(input_msg)

	#vectorize

	vector_input = cv.transform([transformed_msg])

	#predict
 
	result = model.predict(vector_input)[0]

	if result == 1:
		st.header("Spam")
	else:
		st.header("Not Spam")