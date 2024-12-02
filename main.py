import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# data = pd.read_csv('TEST Dataset.csv')
data = pd.read_csv('IMDB Dataset.csv')

reviews = data['review']
sentiments = data['sentiment']

#Removing Punctuation and printing an example
# print("\n PUNCTUATION INTACT: \n", reviews[0])
reviews = reviews.str.lower().str.strip().str.translate(str.maketrans('', '', string.punctuation))
# print("\n PUNCTUATION REMOVED: \n", reviews[0])

#Creating Bag Of Words
vectorizer = CountVectorizer(stop_words='english') #Removes some words like 'the' 'and' 'a' 
x = vectorizer.fit_transform(reviews)

#Predictive Model
x_train, x_test, y_train, y_test = train_test_split(x, sentiments, test_size=0.2, random_state=1)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')



