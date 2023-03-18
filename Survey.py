import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load survey data
survey_data = pd.read_csv("survey_data.csv")


# Define preprocessing functions
def preprocess(text):
    # Perform text preprocessing (e.g., tokenization, stopword removal, stemming, etc.)
    return preprocessed_text


# Preprocess survey responses
survey_data['clean_response'] = survey_data['response'].apply(preprocess)

# Train LDA model
vectorizer = TfidfVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(survey_data['clean_response'])
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(tfidf)


# Extract topics from LDA model
def extract_topics(lda_model, vectorizer, n_top_words):
    words = vectorizer.get_feature_names()
    topics = []
    for i, topic in enumerate(lda_model.components_):
        topic_words = [words[j] for j in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append((i, ', '.join(topic_words)))
    return topics


topics = extract_topics(lda_model, vectorizer, 5)
print(topics)


# Generate summary of survey responses
def generate_summary(response, lda_model, vectorizer, topics):
    # Perform text preprocessing on the response
    cleaned_response = preprocess(response)

    # Convert the response into a vector using TF-IDF
    response_vector = vectorizer.transform([cleaned_response])

    # Use LDA to generate topic distribution for the response
    topic_distribution = lda_model.transform(response_vector)

    # Identify the dominant topic in the response
    dominant_topic = np.argmax(topic_distribution)

    # Retrieve the top words associated with the dominant topic
    topic_words = topics[dominant_topic][1]

    # Generate a summary of the response based on the dominant topic and associated words
    summary = f"The response is mainly about topic {dominant_topic}, which is associated with the following words: {topic_words}."

    return summary


# Test the model
test_responses = [
    "I am satisfied with the quality of service.",
    "The product is overpriced and does not meet my needs.",
    "The customer support was very helpful and responsive.",
    "I would recommend this product to others."
]

for response in test_responses:
    summary = generate_summary(response, lda_model, vectorizer, topics)
    print(summary)
