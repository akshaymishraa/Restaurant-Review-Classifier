from flask import Flask, render_template, request
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
classifier = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		data = request.form['message']
		data = re.sub('[^a-zA-Z]', ' ', data)
		data = data.lower()
		data = data.split()
		all_stopwords = stopwords.words('english')
		all_stopwords.remove('not')
		ps = PorterStemmer()
		data = [ps.stem(word) for word in data if not word in set(all_stopwords)]
		data = ' '.join(data)
		temp = cv.transform([data]).toarray()
		my_prediction = classifier.predict(temp)
		return render_template('result.html', prediction=my_prediction[0])

if __name__ == '__main__':
	app.run(debug=True)