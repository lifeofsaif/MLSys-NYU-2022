from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# Fetch data
data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
texts = data.data
labels = data.target

# Create tf-idf features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, labels)

# Evaluate Classifier
test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))  
X_test = vectorizer.transform(test_data.data)  
y_test = test_data.target  

y_pred = clf.predict(X_test)
  
# Print classification report 
print(classification_report(y_test, y_pred, target_names=test_data.target_names))

# Best performing class: 'talk.politics.mideast' — high precision and recall
# Likely because it uses very specific, topic-heavy language (e.g., "Israel", "Palestine")

# Worst performing class: 'talk.religion.misc' — low recall
# Possibly due to vague, overlapping language with other religion/politics categories