#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import pandas as pd
from langdetect import detect
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from langdetect import DetectorFactory

# Ensure consistent results with langdetect
DetectorFactory.seed = 0

# Sample Data (as if read from CSV)
data = {
    'Text': [
        "Hello, how are you?",
        "Bonjour, comment ça va?",
        "नमस्ते, आप कैसे हैं?",

    ]
}

# Convert the sample data into a DataFrame
df = pd.DataFrame(data)

# Language detection function
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"  # In case the language cannot be detected

# Word count function (tokenizes text into words)
def word_count(text):
    words = word_tokenize(text)
    return len(words)

# Sentence count function (tokenizes text into sentences)
def sentence_count(text):
    sentences = sent_tokenize(text)
    return len(sentences)

# Apply the functions to the dataframe
df['language'] = df['Text'].apply(detect_language)
df['word_count'] = df['Text'].apply(word_count)
df['sentence_count'] = df['Text'].apply(sentence_count)
df['word_tokens'] = df['Text'].apply(lambda x: word_tokenize(x))
df['sentence_tokens'] = df['Text'].apply(lambda x: sent_tokenize(x))

# Display the processed DataFrame (optional)
print(df)


# In[2]:


import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab') # Download the punkt_tab resource

# Create an instance of the SnowballStemmer for German
stemmer = SnowballStemmer("german")

# Accept a sentence input from the user
sentence = input("Enter a German sentence: ")

# Tokenize the sentence into words
words = word_tokenize(sentence)

# Perform stemming for each word in the list
stemmed_words = [stemmer.stem(word) for word in words]

# Output the original and stemmed words
print("\nOriginal and Stemmed Words:")
for original, stemmed in zip(words, stemmed_words):
    print(f"Original: {original}, Stemmed: {stemmed}")





# In[3]:


import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')  # This may not be necessary, but keeping it for completeness

# Accept sentence from the user
sentence = input("Please enter a sentence: ")

# Tokenize the sentence into words
words = word_tokenize(sentence)

# Perform POS tagging
pos_tags = pos_tag(words)

# Print the POS tags
print(pos_tags)


# In[1]:


# Option 2: Word Embeddings (Word2Vec)

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

nltk.download('punkt')

# Sample paragraph
text = """
Natural language processing (NLP) is a field of artificial intelligence. 
It focuses on the interaction between computers and humans through language.
"""

# Tokenize into sentences, then into words
sentences = sent_tokenize(text)
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Access word vector
word = 'language'
if word in model.wv:
    print(f"Vector for '{word}':\n", model.wv[word])
else:
    print(f"'{word}' not found in the vocabulary.")

# Find similar words
print("\nWords similar to 'language':")
print(model.wv.most_similar('language'))


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
documents = ["This is the first document.",
			"This document is the second document.",
			"And this is the third one.",
			"Is this the first document?"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print("Bag-of-Words Matrix:")
print(X.toarray())
print("Vocabulary (Feature Names):", feature_names)





# In[5]:


import nltk

# Download the 'wordnet' resource
nltk.download('wordnet')

# import these modules
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize_word(word, pos=None):
    if pos:
        return lemmatizer.lemmatize(word, pos=pos)
    else:
        return lemmatizer.lemmatize(word)

def main():
    while True:
        # Display the menu
        print("\n--- Lemmatizer Menu ---")
        print("1. Lemmatize a word without POS tag")
        print("2. Lemmatize a word with POS tag (adjective)")
        print("3. Lemmatize a word with POS tag (verb)")
        print("4. Exit")

        choice = input("Enter your choice (1/2/3/4): ")

        if choice == '1':
            word = input("Enter the word to lemmatize: ")
            print(f"Lemmatized word: {lemmatize_word(word)}")

        elif choice == '2':
            word = input("Enter the word to lemmatize: ")
            print(f"Lemmatized word (as adjective): {lemmatize_word(word, pos='a')}")

        elif choice == '3':
            word = input("Enter the word to lemmatize: ")
            print(f"Lemmatized word (as verb): {lemmatize_word(word, pos='v')}")

        elif choice == '4':
            print("Exiting...")
            break

        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()





# In[1]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from collections import Counter

# Download required resources
nltk.download('punkt')
nltk.download('vader_lexicon')  # Download the VADER lexicon
nltk.download('punkt_tab')  # Download the punkt_tab data for tokenization

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Basic emotion dictionary
emotion_dict = {
    'anger': ['angry', 'mad', 'rage'],
    'joy': ['happy', 'joyful', 'excited'],
    'sadness': ['sad', 'unhappy', 'depressed'],
    'fear': ['scared', 'afraid', 'terrified'],
    'surprise': ['surprised', 'shocked', 'amazed'],
    'disgust': ['disgusted', 'gross']
}

# Sentiment Analysis using VADER
def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    return 'POSITIVE' if sentiment_scores['compound'] >= 0.05 else ('NEGATIVE' if sentiment_scores['compound'] <= -0.05 else 'NEUTRAL')

# Emotion Detection based on keywords
def analyze_emotion(text):
    words = word_tokenize(text.lower())
    word_counts = Counter(words)
    emotion_scores = {emotion: sum(word_counts[key] for key in keywords) for emotion, keywords in emotion_dict.items()}
    return max(emotion_scores, key=emotion_scores.get) if max(emotion_scores.values()) > 0 else 'Neutral'

# Function to analyze both sentiment and emotion
def analyze_text(text):
    sentiment = analyze_sentiment(text)
    emotion = analyze_emotion(text)
    return sentiment, emotion

# Menu-driven program
def menu():
    print("\nWelcome to Sentiment and Emotion Analyzer")
    print("1. Analyze Sentiment")
    print("2. Analyze Emotion")
    print("3. Analyze Both Sentiment and Emotion")
    print("4. Exit")

    while True:
        choice = input("\nEnter your choice (1/2/3/4): ")

        if choice == '1':
            text = input("\nEnter text to analyze sentiment: ")
            sentiment = analyze_sentiment(text)
            print(f"Sentiment: {sentiment}")

        elif choice == '2':
            text = input("\nEnter text to analyze emotion: ")
            emotion = analyze_emotion(text)
            print(f"Emotion: {emotion}")

        elif choice == '3':
            text = input("\nEnter text to analyze both sentiment and emotion: ")
            sentiment, emotion = analyze_text(text)
            print(f"Sentiment: {sentiment}, Emotion: {emotion}")

        elif choice == '4':
            print("Exiting the program. Goodbye!")
            break

        else:
            print("Invalid choice. Please select a valid option.")

# Call the menu function to start the program
if __name__ == "__main__":
    menu()




# In[ ]:




