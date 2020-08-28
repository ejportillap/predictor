from flask import Flask, request, jsonify
from gensim.models.keyedvectors import KeyedVectors
import pandas as pd
import numpy as np
import pickle, gcsfs
import util


wordvectors_file_vec = "glove-sbwc.i25.vec"
wordvectors = KeyedVectors.load_word2vec_format(wordvectors_file_vec)
stop_words = pd.read_csv("stop_words.txt")
stop_words = stop_words.words.values
file_name = "model_12.sav"
model = pickle.load(open(file_name, "rb"))


app = Flask(__name__)


@app.route("/findsubjet", methods=["POST"])
def predict():
    req_json = request.json()
    text = req_json["text"]
    vector = util.get_vectors(text, wordvectors)
    labels = model.predict_proba([vector])[0]
    cat = model.predict_proba([vector])[0]
    code, label  = util.get_result(labels, cat)
    response = {"code":code, "label":label}
    return response


@app.route("/update", methods=["POST"])
def update():
    response = request.json()
    path = response["model"]
    project = response["project"]
    bucket = response["bucket"]
    fs = gcsfs.GCSFileSystem(project=project)
    fs.ls(bucket)
    with fs.open(path, 'rb') as file:
        model = pickle.load(file)
    return {"status":"success"}


if __name__ == "__main__":
    app.run(port=5001, debug=True)