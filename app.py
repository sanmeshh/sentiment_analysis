import pickle
import requests
import os

model_url ="https://drive.google.com/file/d/1GEEiCZlolQtijcTmd5Ohk2Q4BZ8tmDr-/view?usp=sharing"
model_path="model.pkl"

def download_model():
    if not os.path.exists(model_path):
        print("Downloading model...")
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded.")

download_model()

with open('model.pkl','rb') as file:
    model=pickle.load(file)

from flask import Flask,render_template,request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


app=Flask(__name__)

with open('tokenizer.pkl','rb') as file:
    tokenizer=pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    txt=request.form['text']
    txt_seq =tokenizer.texts_to_sequences([txt])
    txt_padded=pad_sequences(txt_seq,maxlen=35,padding='pre')

    prediction = model.predict(txt_padded)[0][0]
    if prediction >= 0.5:
        sentiment = f"{prediction} - Positive 👍"
    else:
        sentiment = f"{prediction} - Negative 👎"

    return render_template('index.html', sentiment=sentiment, input_text=txt)

if __name__ == '__main__':
    app.run(debug=True)

