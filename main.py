import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('IMDB Dataset.csv')
reviews = data['review']

#Removing Punctuation and printing an example
print("\n PUNCTUATION INTACT: \n", reviews[0])
reviews = reviews.str.lower().str.strip().str.translate(str.maketrans('', '', string.punctuation))
print("\n PUNCTUATION REMOVED: \n", reviews[0])



