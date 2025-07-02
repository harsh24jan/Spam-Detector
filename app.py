from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')
ps = PorterStemmer()

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
def transform_text(text):
    # Preprocessing same as training
    text = text.lower()
    text = re.findall(r'\b\w+\b', text)
    y = []
    for word in text:
        if word not in stopwords.words('english'):
            y.append(ps.stem(word))
    return " ".join(y)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    input_message = ""

    if request.method == 'POST':
        input_message = request.form.get('message')
        if input_message:
            transformed = transform_text(input_message)
            vector = tfidf.transform([transformed])
            pred = model.predict(vector)[0]
            prediction = "Spam" if pred == 1 else "Not Spam"

    return render_template('index.html', prediction=prediction, message=input_message)

if __name__ == '__main__':
    app.run(debug=True)
