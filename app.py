from flask import Flask, render_template_string, request
import joblib
import pandas as pd
import os
import re
import string
import numpy as np

# Text cleaning as in the notebook
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# No NLTK-based features; just use vectorizer
def extract_ai_features(df, vectorizer=None):
    if vectorizer is None:
        raise ValueError("A fitted vectorizer must be provided for prediction.")
    text_features = vectorizer.transform(df['text'])
    return text_features.toarray()

# Load models and vectorizers as in the notebook
def load_models_from_notebook():
    models_folder = os.path.join(os.getcwd(), 'Models')
    # Load models
    model_fake_news = joblib.load(os.path.join(models_folder, 'best_model_fake_news_tfidf_1.0000.joblib'))
    model_ai = joblib.load(os.path.join(models_folder, 'best_model_ai_generated_content_tfidf_1.0000.joblib'))
    # Load correct vectorizers for each model
    # For MultinomialNB, use count_vectorizer_fake_news.joblib if it exists and matches model
    try:
        count_vectorizer_fake_news = joblib.load(os.path.join(models_folder, 'count_vectorizer_fake_news.joblib'))
        # Check if the number of features matches
        if hasattr(model_fake_news, 'feature_count_') and count_vectorizer_fake_news.vocabulary_:
            if len(count_vectorizer_fake_news.get_feature_names_out()) == model_fake_news.feature_count_.shape[1]:
                vectorizer_fake_news = count_vectorizer_fake_news
            else:
                vectorizer_fake_news = joblib.load(os.path.join(models_folder, 'tfidf_vectorizer_fake_news.joblib'))
        else:
            vectorizer_fake_news = joblib.load(os.path.join(models_folder, 'tfidf_vectorizer_fake_news.joblib'))
    except Exception:
        vectorizer_fake_news = joblib.load(os.path.join(models_folder, 'tfidf_vectorizer_fake_news.joblib'))

    vectorizer_ai = joblib.load(os.path.join(models_folder, 'tfidf_vectorizer_ai_generated_content.joblib'))
    return model_fake_news, model_ai, vectorizer_fake_news, vectorizer_ai

# Predict using notebook logic (no NLTK features)
def predict_text(text, model, vectorizer, task):
    processed_text = clean_text(text)
    if task == 'AI Generated Content':
        df = pd.DataFrame({'text': [processed_text]})
        vectorized_text = extract_ai_features(df, vectorizer=vectorizer)
    else:
        vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0][1]
    return prediction, probability

app = Flask(__name__)
model_fake_news, model_ai, vectorizer_fake_news, vectorizer_ai = load_models_from_notebook()

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
        fake_news_pred, fake_news_prob = predict_text(user_text, model_fake_news, vectorizer_fake_news, 'Fake News')
        ai_content_pred, ai_content_prob = predict_text(user_text, model_ai, vectorizer_ai, 'AI Generated Content')
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
