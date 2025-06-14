from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
import pandas as pd
import pickle
import base64
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)
STOPWORDS = set(stopwords.words("english"))

# Load models once at startup
try:
    with open("Models/model_xgb.pkl", "rb") as f:
        predictor = pickle.load(f)
    with open("Models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("Models/countVectorizer.pkl", "rb") as f:
        cv = pickle.load(f)
    
    # Check if the model has predict_proba method
    if not hasattr(predictor, 'predict_proba'):
        print("Warning: Model does not have predict_proba method. Using decision tree model as fallback.")
        # Try to load decision tree model as fallback
        with open("Models/model_dt.pkl", "rb") as f:
            predictor = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

def preprocess_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    words = [stemmer.stem(word) for word in text.lower().split() if word not in STOPWORDS]
    return " ".join(words)

@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)
            data["Predicted"] = data["Sentence"].apply(lambda x: predict_sentiment(x))
            
            # Generate graph
            plt.figure(figsize=(5,5))
            data["Predicted"].value_counts().plot(kind="pie", autopct="%1.1f%%")
            img = BytesIO()
            plt.savefig(img, format="png")
            plt.close()
            img.seek(0)
            
            # Prepare response
            output = BytesIO()
            data.to_csv(output, index=False)
            output.seek(0)
            
            response = send_file(output, mimetype="text/csv", 
                               as_attachment=True, download_name="predictions.csv")
            response.headers["X-Graph"] = base64.b64encode(img.getvalue()).decode("utf-8")
            return response
        
        elif request.json and "text" in request.json:
            text = request.json["text"]
            return jsonify({"result": predict_sentiment(text)})
        elif request.form and "text" in request.form:
            text = request.form["text"]
            return jsonify({"result": predict_sentiment(text)})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def predict_sentiment(text):
    # If text is empty, return a default value
    if not text or text.strip() == "":
        return "Neutral"
    
    # Preprocess the text
    processed = preprocess_text(text)
    
    # If processed text is empty (e.g., only stopwords or non-alphabetic characters)
    if not processed or processed.strip() == "":
        return "Neutral"
    
    # Vectorize and scale the text
    vectorized = cv.transform([processed]).toarray()
    scaled = scaler.transform(vectorized)
    
    try:
        # Get prediction and probability
        prediction = predictor.predict(scaled)[0]
        probabilities = predictor.predict_proba(scaled)[0]
        
        # Log for debugging
        print(f"Input text: {text}")
        print(f"Processed text: {processed}")
        print(f"Prediction value: {prediction}")
        print(f"Prediction probabilities: {probabilities}")
        
        # Check if we have probabilities for multiple classes
        if len(probabilities) > 1:
            # Get probabilities for each class
            class_0_prob = probabilities[0]  # Typically negative class
            class_1_prob = probabilities[1]  # Typically positive class
            
            # Use a threshold to determine sentiment
            # If the model is biased, we can adjust the threshold
            if class_1_prob > 0.6:  # Higher threshold for positive class
                return "Positive"
            elif class_0_prob > 0.5:  # Standard threshold for negative class
                return "Negative"
            else:
                # If neither probability is strong enough, return neutral
                return "Neutral"
        else:
            # If we only have one probability, use the prediction directly
            return "Positive" if prediction == 1 else "Negative"
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # If there's an error, return a default value
        return "Neutral"

if __name__ == "__main__":
    app.run(port=5000, debug=True)