from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

import pandas as pd
df = pd.read_pickle("dataset/intents_asli.pkl")

app = Flask(__name__)
CORS(app)

@app.before_first_request
def load_models():
    global df
    df = pd.read_pickle("dataset/intents_asli.pkl")

@app.route("/ask_chatbot", methods=['GET', 'POST'])
def print_sentences():
    if request.method == "POST":
        import warnings
        warnings.filterwarnings("ignore") # untuk menyembunyikan warning

        data = request.get_json()
        sentences = data["sentences"]
        
        # stemming
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        sentences = StemmerFactory().create_stemmer().stem(sentences)

        import pandas as pd
        df = pd.read_pickle("dataset/intents_asli.pkl")

        def predict_response(nama_model, sentences):
            import pickle
            tf_idf = pickle.load(open("word_embedding/"+nama_model+".pkl", 'rb'))
            X_test_vecs = tf_idf.transform([sentences]).todense().tolist()

            intent_classifier = pickle.load(open("intent_classification_model/"+nama_model+".pkl", 'rb'))
            intent_predict = intent_classifier.predict(X_test_vecs)
            intent_predict = intent_predict[0]

            temp_df = df[df['intents'] == intent_predict]

            if len(temp_df) == 0:
                # masih belum ditemukan
                nama_model = intent_predict
                return predict_response(nama_model, sentences)
            else:
                temp_df = temp_df.reset_index(drop=True)
                confidence_scores = max(intent_classifier.predict_proba(X_test_vecs)[0])
                return temp_df["text_output"][0], confidence_scores

        response, conf_score = predict_response("root", sentences)

        if conf_score > 0.3:
            import csv
            with open('chatbot_log.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                rows = [[data["username"], sentences, response]]
                writer.writerows(rows)
            print("Save Log Chat")
            
            json_data = {
                "success": 1,
                "message": "Get response successfully. Confidence Score = " + str(conf_score),
                "data": response
            }
            return jsonify(json_data), 201
        else:
            json_data = {
                "success": 0,
                "message": "Get response failed. Confidence Score = " + str(conf_score),
                "data": "Maaf saya masih ragu " + "{:.0%}".format(1 - conf_score) + " dengan maksud anda dan mohon menggunakan bahasa Indonesia yang baku."
            }
            return jsonify(json_data), 201

    json_data = {
        "success": 0,
        "message": "Get response failed",
        "data": "Maaf saya belum mengerti maksud anda dan mohon menggunakan bahasa Indonesia yang baku."
    }
    return jsonify(json_data), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)