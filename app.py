import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__, static_url_path='/static')
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vector.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    # features = [ for x in ]
    author = request.form.get('author')
    news = request.form.get('news')
    content = author + ' ' + news
    port_stem = PorterStemmer()
    content = re.sub('[^a-zA-Z]',' ', content) # Exclude a-z or A-Z and substitute rest with ' '
    content = content.lower()
    content = content.split() # Convert into list
    content = [port_stem.stem(word) for word in content if not word in stopwords.words("english")] # Filter stopwords
    content = ' '.join(content)
    content = vectorizer.transform([content])
    final_features = np.array(content).reshape(1, -1)
    prediction = model.predict(content)

    if prediction == 0:
        output = 'True'
    else:
        output = 'Fake'

    return render_template('index.html', prediction_text='The News is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
