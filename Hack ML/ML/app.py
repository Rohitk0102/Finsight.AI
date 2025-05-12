# app.py
import os
import requests
from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv  # Can be removed if not loading any keys from .env
import logging
from typing import Dict, List, Union, Any, Optional
import traceback  # For better error logging
import datetime  # For parsing dates

# --- Configuration & Initialization ---
logging.basicConfig(level=logging.INFO)  # Basic logging

# --- API Keys & URLs ---
MARKETAUX_API_KEY = "G4qN7vEnOdTBRN4MuLAZbXOg7q93fFV9AlB9J38x"
MARKETAUX_NEWS_URL = "https://api.marketaux.com/v1/news/all"
MARKETAUX_WEBSITE_URL = "https://www.marketaux.com/"

app = Flask(__name__)

# --- ML Model Loading ---
summarizer = None
sentiment_analyzer = None
models_loaded = False
tokenizer = None

print("Loading ML models...")
try:
    summarizer_model_name = "google-t5/t5-small"
    sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    sentiment_analyzer = pipeline(
        "sentiment-analysis", model=sentiment_model_name)
    print("ML models loaded successfully.")
    models_loaded = True
except Exception as e:
    print(f"CRITICAL: Error loading ML models: {e}")
    traceback.print_exc()
    models_loaded = False


# --- Helper Functions ---

def fetch_marketaux_news(query: str, page: int = 1, articles_per_page: int = 10) -> Dict[str, Any]:
    """Fetches news articles from MarketAux API based on a search query and page number."""
    if not MARKETAUX_API_KEY:
        app.logger.error("MARKETAUX_API_KEY not configured.")
        return {"error": "Server configuration error: Missing MarketAux API Key."}

    search_term = query.strip()
    app.logger.info(
        f"Fetching news from MarketAux for query: '{search_term}', page: {page}, limit: {articles_per_page}")

    try:
        params = {
            'api_token': MARKETAUX_API_KEY,
            'search': search_term,
            'language': 'en',
            'limit': articles_per_page,
            'page': page,
            'sort': 'published_on',
        }

        masked_params = params.copy()
        masked_params['api_token'] = '***'
        request_url = requests.Request(
            'GET', MARKETAUX_NEWS_URL, params=masked_params).prepare().url
        app.logger.info(f"Requesting URL (token masked): {request_url}")

        response = requests.get(MARKETAUX_NEWS_URL, params=params, timeout=15)
        app.logger.info(
            f"MarketAux API Response Status Code: {response.status_code}")
        response.raise_for_status()
        data = response.json()

        if 'data' in data and isinstance(data['data'], list):
            raw_articles = data.get("data", [])
            app.logger.info(
                f"Fetched {len(raw_articles)} articles from MarketAux (Page {page}).")
            if not raw_articles and page == 1:
                app.logger.warning(
                    f"MarketAux returned 0 results for query '{search_term}'.")

            processed_articles = []
            for article in raw_articles:
                content = article.get('description') or article.get(
                    'snippet') or article.get('title', '')
                if not isinstance(content, str):
                    content = str(content) if content is not None else ''

                published_dt_str = "N/A"
                published_at = article.get('published_at')
                if published_at:
                    try:
                        dt_obj = datetime.datetime.fromisoformat(
                            published_at.replace('Z', '+00:00'))
                        published_dt_str = dt_obj.strftime(
                            '%Y-%m-%d %H:%M:%S UTC')
                    except ValueError:
                        app.logger.warning(
                            f"Could not parse MarketAux date: {published_at}")
                        published_dt_str = published_at

                processed_articles.append({
                    'title': article.get('title', 'No Title'),
                    'link': article.get('url', '#'),
                    'source': article.get('source', 'Unknown Source'),
                    'published_utc': published_dt_str,
                    'content_for_analysis': content
                })
            return {"articles": processed_articles}
        elif 'error' in data and isinstance(data['error'], dict):
            error_details = data['error']
            error_message = error_details.get(
                "message", "Unknown API error from MarketAux")
            error_code = error_details.get("code", "N/A")
            app.logger.error(
                f"MarketAux API error: Code={error_code}, Message={error_message}")
            return {"error": f"MarketAux News API Error ({error_code}): {error_message}"}
        else:
            app.logger.error(
                f"Unexpected response format from MarketAux: {data}")
            return {"error": "Unexpected response format received from MarketAux."}

    except requests.exceptions.HTTPError as http_err:
        app.logger.error(
            f"HTTP error fetching news from MarketAux: {http_err}")
        try:
            error_details = response.json().get('error', {})
            error_msg = error_details.get(
                'message', f"Status {response.status_code}")
        except:
            error_msg = f"Status {response.status_code}, Body: {response.text[:200]}"
        return {"error": f"HTTP error connecting to MarketAux: {error_msg}"}
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Network error fetching news from MarketAux: {e}")
        return {"error": f"Network or API request error: {e}"}
    except Exception as e:
        app.logger.error(f"Unexpected error fetching news: {e}")
        traceback.print_exc()
        return {"error": f"An unexpected error occurred: {e}"}


def analyze_content(text: str) -> Dict[str, Any]:
    """Analyzes text for summary and sentiment using loaded models."""
    if not models_loaded or summarizer is None or sentiment_analyzer is None:
        return {"error": "ML models not available."}
    if not text or not isinstance(text, str):
        return {"summary": "No content provided for analysis.", "sentiment": "N/A", "sentiment_score": None}

    analysis = {}
    try:
        # Summarization
        inputs = tokenizer(text, return_tensors="pt",
                           max_length=512, truncation=True, padding=True)
        decoded_input = tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True)
        summary_result = summarizer(
            decoded_input,
            max_length=100,
            min_length=20,
            do_sample=False
        )[0]
        analysis['summary'] = summary_result['summary_text']
    except Exception as e:
        app.logger.error(f"Error during summarization: {e}")
        analysis['summary'] = "Error generating summary."

    try:
        # Sentiment Analysis
        sentiment_result = sentiment_analyzer(text[:512])[0]
        analysis['sentiment'] = sentiment_result['label']
        # We are not using the score anymore, but we still get it from the model
        # analysis['sentiment_score'] = round(sentiment_result['score'], 3)
    except Exception as e:
        app.logger.error(f"Error during sentiment analysis: {e}")
        analysis['sentiment'] = "Error analyzing sentiment."
        # analysis['sentiment_score'] = None

    return analysis

# --- HTML Templates (as strings) ---


HTML_FORM_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Analysis</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background-color: #f4f4f4; }
        .container { background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); max-width: 600px; margin: 40px auto;}
        h1 { color: #333; text-align: center; margin-bottom: 20px;}
        label { font-weight: bold; margin-bottom: 5px; display: block; }
        input[type=text] { width: 100%; padding: 10px; margin-bottom: 15px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        button { background-color: #3498db; color: white; padding: 12px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; width: 100%; } /* MarketAux Blue */
        button:hover { background-color: #2980b9; }
        .error { color: red; margin-top: 10px; }
        .footer { margin-top: 30px; text-align: center; font-size: 0.9em; color: #777; }
        a { color: #3498db; text-decoration: none; } /* MarketAux Blue */
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>News Analysis</h1>
        <form action="/analyze" method="POST">
            <div>
                <label for="topic">Enter News Topic or Ticker Symbol:</label>
                <input type="text" id="topic" name="topic" required placeholder="e.g., AAPL, Tesla, AI regulation...">
            </div>
            <input type="hidden" name="page" value="1">
            <div>
                <button type="submit">Analyze News</button>
            </div>
        </form>
        {% if error %}
            <p class="error">Error: {{ error }}</p>
        {% endif %}
    </div>
    <div class="footer">
        News potentially via <a href="{{ marketaux_url }}" target="_blank">MarketAux.com</a> | Analysis by Hugging Face models.
    </div>
</body>
</html>
"""

# *** MODIFIED HTML_RESULTS_TEMPLATE below ***
HTML_RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Analysis Results for {{ topic }}</title>
     <style>
        body { font-family: sans-serif; margin: 20px; background-color: #f4f4f4; }
        .container { background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; max-width: 800px; margin: 40px auto;}
        h1, h2 { color: #333; text-align: center; margin-bottom: 20px;}
        h2 { color: #2980b9; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; margin-bottom: 15px; text-align: left;} /* MarketAux Blue */
        .article { border: 1px solid #eee; padding: 15px; margin-bottom: 15px; border-radius: 5px; background-color: #fdfdfd;}
        .article-title { font-weight: bold; font-size: 1.1em; margin-bottom: 5px;}
        .article-meta { font-size: 0.85em; color: #555; margin-bottom: 10px; }
        .article-link { font-size: 0.9em; color: #3498db; text-decoration: none; word-break: break-all;} /* MarketAux Blue */
        .article-link:hover { text-decoration: underline; }
        .analysis { background-color: #eaf6ff; padding: 10px; margin-top: 10px; border-radius: 4px; border-left: 3px solid #3498db; } /* MarketAux Blue */
        .analysis p { margin: 5px 0; font-size: 0.95em;}
        .error { color: red; font-weight: bold; }
        .footer { margin-top: 30px; text-align: center; font-size: 0.9em; color: #777; }
        a { color: #3498db; text-decoration: none; } /* MarketAux Blue */
        a:hover { text-decoration: underline; }
        /* Styles for new search form and load more */
        .search-again-form { margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; }
        .search-again-form label { font-weight: bold; margin-bottom: 5px; display: block; }
        .search-again-form input[type=text] { width: calc(100% - 120px); padding: 10px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; display: inline-block; vertical-align: middle; }
        .search-again-form button { background-color: #e67e22; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; width: 100px; display: inline-block; vertical-align: middle; margin-left: 10px;} /* Orange */
        .search-again-form button:hover { background-color: #d35400; }
        .load-more-form { text-align: center; margin-top: 20px; }
        .load-more-form button { background-color: #2ecc71; color: white; padding: 10px 25px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; } /* Green */
        .load-more-form button:hover { background-color: #27ae60; }
        /* Styles for sentiment colors */
        .effect-positive { color: #27ae60; font-weight: bold; } /* Green */
        .effect-negative { color: #e74c3c; font-weight: bold; } /* Red */
        .effect-neutral { color: #f39c12; font-weight: bold; } /* Orange */
     </style>
</head>
<body>
    <div class="container">
        <h1>News Analysis Results for "{{ topic }}"</h1>
        <h2>Page {{ current_page }}</h2>

        {% if error %}
            <p class="error">Error fetching or processing news: {{ error }}</p>
        {% elif articles %}
            {% for article in articles %}
            <div class="article">
                <p class="article-title">{{ article.title }}</p>
                <p class="article-meta">Source: {{ article.source }} | Published: {{ article.published_utc }}</p>
                <p><a href="{{ article.link }}" target="_blank" class="article-link">{{ article.link }}</a></p>
                {% if article.analysis and not article.analysis.get('error') %}
                    <div class="analysis">
                        {# Changed Summary Label #}
                        <p><strong>AI Overview:</strong> {{ article.analysis.summary }}</p>
                        {# Changed Sentiment Label and added color logic #}
                        <p><strong>Market Effect:</strong>
                            {% set sentiment = article.analysis.sentiment %}
                            {% if sentiment == 'POSITIVE' %}
                                <span class="effect-positive">Positive Effect</span>
                            {% elif sentiment == 'NEGATIVE' %}
                                <span class="effect-negative">Negative Effect</span>
                            {% else %}
                                {# Handle neutral or unexpected values #}
                                <span class="effect-neutral">Neutral Effect</span>
                            {% endif %}
                        </p>
                    </div>
                {% elif article.analysis and article.analysis.get('error') %}
                     <p class="error">Analysis Error: {{ article.analysis.error }}</p>
                {% else %}
                    <p class="error">Could not analyze this article (perhaps no summary/description content).</p>
                {% endif %}
                </div>
            {% endfor %}

            {% if articles %} {# Only show Load More if current page had results #}
            <div class="load-more-form">
                <form action="/analyze" method="POST">
                    <input type="hidden" name="topic" value="{{ topic }}">
                    <input type="hidden" name="page" value="{{ next_page }}">
                    <button type="submit">Load More Articles (Page {{ next_page }})</button>
                </form>
            </div>
            {% endif %}

        {% else %}
             <p>No news articles found for the specified topic/ticker via MarketAux.com (Page {{ current_page }}).</p>
        {% endif %}

        <div class="search-again-form">
             <h2>Search Again</h2>
             <form action="/analyze" method="POST">
                 <div>
                     <label for="new_topic">Enter New Topic or Ticker Symbol:</label>
                     <input type="text" id="new_topic" name="topic" required placeholder="e.g., GOOG, Microsoft, interest rates...">
                     <input type="hidden" name="page" value="1">
                     <button type="submit">Search</button>
                 </div>
             </form>
        </div>

        <hr style="margin-top: 30px;">
        <p style="text-align: center;"><a href="/">Back to Home Page</a></p>
    </div>

    <div class="footer">
        News potentially via <a href="{{ marketaux_url }}" target="_blank">MarketAux.com</a> | Analysis by Hugging Face models.
    </div>
</body>
</html>
"""

# --- Flask Routes ---


@app.route('/', methods=['GET'])
def home():
    """Displays the input form."""
    error_msg = None
    if not models_loaded:
        error_msg = "ML Models failed to load. Analysis functionality is disabled."
    return render_template_string(HTML_FORM_TEMPLATE, error=error_msg, marketaux_url=MARKETAUX_WEBSITE_URL)

# Updated route to handle page number


@app.route('/analyze', methods=['POST'])
def analyze_news_route():
    """Handles form submission, fetches news from MarketAux, analyzes, and displays results."""
    if not models_loaded:
        return render_template_string(HTML_FORM_TEMPLATE, error="ML Models are not available. Check server logs.")

    query_input = request.form.get('topic')
    try:
        current_page = int(request.form.get('page', 1))
        if current_page < 1:
            current_page = 1
    except ValueError:
        current_page = 1

    if not query_input:
        return render_template_string(HTML_FORM_TEMPLATE, error="Please enter a news topic or ticker.", marketaux_url=MARKETAUX_WEBSITE_URL)

    articles_per_page = 10
    news_result = fetch_marketaux_news(
        query_input,
        page=current_page,
        articles_per_page=articles_per_page
    )

    if "error" in news_result:
        return render_template_string(
            HTML_RESULTS_TEMPLATE,
            topic=query_input,
            error=news_result["error"],
            marketaux_url=MARKETAUX_WEBSITE_URL,
            current_page=current_page,
            next_page=current_page + 1
        )

    processed_articles = []
    articles = news_result.get("articles", [])

    for article in articles:
        content_to_analyze = article.get('content_for_analysis', '')
        if content_to_analyze:
            analysis = analyze_content(content_to_analyze)
            article['analysis'] = analysis
        else:
            article['analysis'] = {
                "error": "No description/snippet available from MarketAux for analysis."}
        processed_articles.append(article)

    next_page = current_page + 1

    return render_template_string(
        HTML_RESULTS_TEMPLATE,
        topic=query_input,
        articles=processed_articles,
        marketaux_url=MARKETAUX_WEBSITE_URL,
        current_page=current_page,
        next_page=next_page
    )


# --- Main Execution ---
if __name__ == '__main__':
    if not models_loaded:
        logging.warning(
            "ML Models failed to load. The /analyze endpoint will return errors.")
    app.run(host='0.0.0.0', port=5001, debug=True)
