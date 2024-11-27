from flask import Flask, render_template, request
import pickle
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os

# Download required NLTK resources if not already present
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Download resources (only if needed)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) - set(["not"])


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    # Preserve "not" with the next word (e.g., "not good")
    tokens = [tokens[i] + "_" + tokens[i + 1] if tokens[i] == "not" and i + 1 < len(tokens) else tokens[i] for i in range(len(tokens))]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and TF-IDF vectorizer
try:
    model = pickle.load(open("sentiment_model (1).pkl", "rb"))
    tfidf_vectorizer = pickle.load(open("tfidf_vectorizer (1).pkl", "rb"))
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    exit()

@app.route("/")
def index():
    """
    Render the main page.
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the sentiment based on user input.
    """
    try:
        if request.method == "POST":
            # Get user input
            review = request.form["review"]

            # Preprocess the review
            processed_review = preprocess_text(review)

            # Transform and predict
            transformed_review = tfidf_vectorizer.transform([processed_review])
            prediction = model.predict(transformed_review)

            # Convert prediction to sentiment
            sentiment = {0: "Negative", 1: "Positive"}
            result = sentiment[prediction[0]]

            return render_template("index.html", review=review, result=result)
    except Exception as e:
        return render_template(
            "index.html", review="Error", result=f"An error occurred: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
