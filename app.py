from flask import Flask, render_template_string, request
import joblib
import pandas as pd
import os
import re
import string
import numpy as np
from collections import Counter
import math
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ", text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text

def sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    number_of_sentences = len(sentences)
    total_words = sum(len(sentence.split()) for sentence in sentences)
    avg_sentence = total_words / number_of_sentences if number_of_sentences > 0 else 0
    return number_of_sentences, avg_sentence

def repetitive_words(text):
    tokens = nltk.word_tokenize(text.lower())
    synsets = [wn.synsets(token) for token in tokens]
    synonyms = [[lemma.name() for synset in token_synsets for lemma in synset.lemmas()] for token_synsets in synsets]
    repeat = sum(len(set(s1) & set(s2)) > 0 for i, s1 in enumerate(synonyms) for s2 in synonyms[i+1:])
    return repeat / len(tokens) if tokens else 0

def entropy(text):
    tokens = nltk.word_tokenize(text.lower())
    token_counts = Counter(tokens)
    total = len(tokens)
    probs = [count / total for count in token_counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def count_punctuation(text):
    sentences = nltk.sent_tokenize(text)
    number_of_sentences = len(sentences)
    count = sum(1 for char in text if char in string.punctuation)
    return count / number_of_sentences if number_of_sentences > 0 else 0

def count_numbers(text):
    sentences = nltk.sent_tokenize(text)
    number_of_sentences = len(sentences)
    count = sum(1 for word in text.split() if any(c.isdigit() for c in word))
    return count / number_of_sentences if number_of_sentences > 0 else 0

def extract_ai_features(df, feature_type='tfidf', vectorizer=None):
    if vectorizer is None:
        raise ValueError("A fitted vectorizer must be provided for prediction.")
    text_features = vectorizer.transform(df['text'])
    df['Sentence_length'], df['Average_sentence_length'] = zip(*df['text'].apply(sentence_length))
    df['Repetitive_words'] = df['text'].apply(repetitive_words)
    df['Entropy'] = df['text'].apply(entropy)
    df['Punctuation_count'] = df['text'].apply(count_punctuation)
    df['Numbers_count'] = df['text'].apply(count_numbers)
    additional_features = df[['Sentence_length', 'Average_sentence_length', 'Repetitive_words', 'Entropy', 'Punctuation_count', 'Numbers_count']]
    combined_features = np.hstack((text_features.toarray(), additional_features))
    return combined_features, vectorizer

def load_best_models():
    models_folder = os.path.join(os.getcwd(), 'Models')
    fake_news_models = [f for f in os.listdir(models_folder) if f.startswith('best_model_fake_news') and f.endswith('.joblib')]
    ai_content_models = [f for f in os.listdir(models_folder) if f.startswith('best_model_ai_generated_content') and f.endswith('.joblib')]
    def get_accuracy(filename):
        return float(filename.split('_')[-1].replace('.joblib', ''))
    best_fake_news_model = max(fake_news_models, key=get_accuracy)
    best_ai_content_model = max(ai_content_models, key=get_accuracy)
    fake_news_model = joblib.load(os.path.join(models_folder, best_fake_news_model))
    ai_content_model = joblib.load(os.path.join(models_folder, best_ai_content_model))
    fake_news_vectorizer = joblib.load(os.path.join(models_folder, 'tfidf_vectorizer_fake_news.joblib'))
    ai_content_vectorizer = joblib.load(os.path.join(models_folder, 'tfidf_vectorizer_ai_generated_content.joblib'))
    return fake_news_model, ai_content_model, fake_news_vectorizer, ai_content_vectorizer

def predict_text(text, model, vectorizer, task):
    processed_text = wordopt(text)
    if task == 'AI Generated Content':
        df = pd.DataFrame({'text': [processed_text]})
        vectorized_text, _ = extract_ai_features(df, vectorizer=vectorizer)
    else:
        vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0][1]
    return prediction, probability

app = Flask(__name__)
fake_news_model, ai_content_model, fake_news_vectorizer, ai_content_vectorizer = load_best_models()

HTML = '''
<!doctype html>
<title>Fake News & AI Content Detector</title>
<h2>Fake News & AI-Generated Content Detector</h2>
<form method=post>
  <textarea name=text rows=8 cols=80 placeholder="Enter your news/article text here..."></textarea><br>
  <input type=submit value=Analyze>
</form>
{% if results %}
  <h3>Results:</h3>
  <ul>
    <li><b>Fake News Detection:</b> {{ results.fake_news_label }} (Confidence: {{ results.fake_news_confidence }})</li>
    <li><b>AI-Generated Content:</b> {{ results.ai_content_label }} (Confidence: {{ results.ai_content_confidence }})</li>
  </ul>
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        user_text = request.form['text']
        fake_news_pred, fake_news_prob = predict_text(user_text, fake_news_model, fake_news_vectorizer, 'Fake News')
        ai_content_pred, ai_content_prob = predict_text(user_text, ai_content_model, ai_content_vectorizer, 'AI Generated Content')
        # Fake News Detection output: 0=Fake, 1=Real
        fake_news_label = 'Fake' if fake_news_pred == 0 else 'Real'
        # AI Generated Content output: 0=AI-generated, 1=Human-written
        ai_content_label = 'AI-generated' if ai_content_pred == 0 else 'Human-written'
        results = {
            'fake_news_label': fake_news_label,
            'fake_news_confidence': f"{fake_news_prob:.2%}",
            'ai_content_label': ai_content_label,
            'ai_content_confidence': f"{ai_content_prob:.2%}"
        }
    return render_template_string(HTML, results=results)

if __name__ == '__main__':
    app.run(debug=True)
