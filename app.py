from flask import Flask,render_template,request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app=Flask(__name__)

with open('model.pkl','rb') as file:
    model=pickle.load(file)

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

