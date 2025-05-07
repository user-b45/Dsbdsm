import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download NLTK data (run only once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
# Sample Document
document = """Text analytics refers to the process of extracting meaningful insights from unstructured text data. 
It includes tasks such as preprocessing, sentiment analysis, and document classification."""

print("📄 Original Document:\n", document, "\n")

# Tokenization
tokens = word_tokenize(document)
print("🔹 Tokenized Words:\n", tokens, "\n")

# POS Tagging
pos_tags = nltk.pos_tag(tokens)
print("🔹 Part-of-Speech Tags:\n", pos_tags, "\n")

# Stop Words Removal
stop_words = set(stopwords.words("english"))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print("🔹 After Stop Words Removal:\n", filtered_tokens, "\n")

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
print("🔹 Stemmed Words:\n", stemmed_words, "\n")

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in filtered_tokens]
print("🔹 Lemmatized Words:\n", lemmatized_words, "\n")

# Document Representation - TF
print("📊 Term Frequency (TF):")
tf_vectorizer = CountVectorizer()
tf_matrix = tf_vectorizer.fit_transform([document])
tf_df = pd.DataFrame(tf_matrix.toarray(), columns=tf_vectorizer.get_feature_names_out())
print(tf_df, "\n")

# Document Representation - TF-IDF
print("📊 TF-IDF:")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([document])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(tfidf_df)

