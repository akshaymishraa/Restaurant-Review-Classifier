# Importing essential libraries
import numpy as np
import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv('Restaurant_reviews.tsv', delimiter='\t', quoting=3)

# Importing essential libraries for performing Natural Language Processing on 'Restaurant_Reviews.tsv' dataset
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Cleaning the reviews
corpus = []
for i in range(0,1000):

  # Cleaning special character from the reviews
  review = re.sub('[^a-zA-Z]',' ',df['Review'][i])

  # Converting the entire review into lower case
  review = review.lower()

  # Tokenizing the review by words
  review = review.split()

  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  # Removing the stop words
  # Stemming the words
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]

  # Joining the stemmed words
  review = ' '.join(review)

  # Creating a corpus
  corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values

pickle.dump(cv, open('cv-transform.pkl', 'wb'))

"""**Model Building**"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha = 0.2)
classifier.fit(X_train, y_train)

pickle.dump(classifier, open('model.pkl', 'wb'))
