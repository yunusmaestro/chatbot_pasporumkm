import asyncio
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
import pickle
import csv
import warnings
from datetime import datetime
import os.path

# Nama file CSV
log_filename = 'chatbot_log.csv'

import pandas as pd
df = pd.read_pickle("dataset/list_jawaban.pkl")

app = Flask(__name__)
CORS(app)

async def predict_response(nama_model, sentences):
    print("Model = ", nama_model)

    tf_idf = pickle.load(open("word_embedding/"+nama_model+".pkl", 'rb'))
    X_test_vecs = tf_idf.transform([sentences]).todense().tolist()

    intent_classifier = pickle.load(open("intent_classification_model/"+nama_model+"_mlp.pkl", 'rb'))
    intent_predict = intent_classifier.predict(X_test_vecs)
    intent_predict = intent_predict[0]

    temp_df = df[df['intents'] == intent_predict]

    if len(temp_df) == 1:
        temp_df = temp_df.reset_index(drop=True)
        confidence_scores = max(intent_classifier.predict_proba(X_test_vecs)[0])
        return temp_df["text_output"][0], confidence_scores, temp_df["intents"][0]
    else:
        # masih belum ditemukan
        nama_model = intent_predict
        # cari dulu di dataframe apakah hanya 1 intent?
        temp_df = df[df['intents'].str.startswith(intent_predict)]
        if len(temp_df) == 1:
            temp_df = temp_df.reset_index(drop=True)
            confidence_scores = max(intent_classifier.predict_proba(X_test_vecs)[0])
            return temp_df["text_output"][0], confidence_scores, temp_df["intents"][0]

        return await predict_response(nama_model, sentences)

async def get_response(data):
    warnings.filterwarnings("ignore") # untuk menyembunyikan warning

    sentences = data["sentences"]

    # stemming
    sentences = StemmerFactory().create_stemmer().stem(sentences)
    print("After Stemming = ", sentences)

    response, conf_score, intents = await predict_response("root", sentences)

    if conf_score > 0.3:
        with open(log_filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            rows = [
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    data["username"],
                    data["sentences"],
                    intents,
                    response
                ]
            ]
            writer.writerows(rows)
        print("Save Log Chat")
        
        json_data = {
            "success": 1,
            "message": "Get response successfully. Confidence Score = " + str(conf_score),
            "data": response
        }
        return jsonify(json_data), 201
    else:
        response = "Maaf saya masih tidak yakin dengan maksud anda dan mohon menggunakan bahasa Indonesia yang baku."
        with open(log_filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            rows = [
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    data["username"],
                    data["sentences"],
                    intents,
                    response
                ]
            ]
            writer.writerows(rows)
        print("Save Log Chat")

        json_data = {
            "success": 0,
            "message": "Get response failed. Confidence Score = " + str(conf_score),
            "data": response
        }
        return jsonify(json_data), 201

@app.route("/ask_chatbot", methods=['GET', 'POST'])
async def print_sentences():
    # Mengecek apakah file CSV sudah ada
    file_exists = os.path.isfile(log_filename)
    if not file_exists:
        with open(log_filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Baris header pada file CSV
            header = ["Timestamp", "Username", "Question", "Intent", "Response"]
            writer.writerow(header)

    if request.method == "POST":
        data = request.get_json()
        response = await get_response(data)
        return response

    json_data = {
        "success": 0,
        "message": "Get response failed",
        "data": response
    }
    return jsonify(json_data), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)