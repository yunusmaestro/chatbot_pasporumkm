from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/ask_chatbot", methods=['GET', 'POST'])
def print_sentences():
    if request.method == "POST":
        import warnings
        warnings.filterwarnings("ignore") # untuk menyembunyikan warning

        # 1. ======================================================================================
        print("GET QUESTION")
        # sentences = "Yang disebut syubhat bahan yang bagaimana?"

        # print(request.form)
        data = request.get_json()

        sentences = data["sentences"]
        
        print("QUESTION = ", sentences)

        print("TEXT PREPROCESSING")
        # lowercase
        sentences = sentences.lower()
        # remove punctuation
        import string
        exclude = set(string.punctuation)
        sentences = ''.join(punct for punct in sentences if punct not in exclude)

        # stemming
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        Fact = StemmerFactory()
        Stemmer = Fact.create_stemmer()
        sentences = Stemmer.stem(sentences)

        print("sentences = ", sentences)

        # 2. ======================================================================================

        import pandas as pd
        df = pd.read_pickle("dataset/intents_asli.pkl")

        def predict_response(nama_model, sentences):
            # print("LOAD TF-IDF EMBEDDING")
            import pickle
            tf_idf = pickle.load(open("word_embedding/"+nama_model+".pkl", 'rb'))
            X_test_vecs = tf_idf.transform([sentences]).todense().tolist()

            # print("CLASSIFICATION")
            intent_classifier = pickle.load(open("intent_classification_model/"+nama_model+".pkl", 'rb'))
            intent_predict = intent_classifier.predict(X_test_vecs)
            intent_predict = intent_predict[0]
            # print(intent_predict)

            # cari di dataframe
            temp_df = df.copy()
            query = "intents == '"+intent_predict+"'"
            temp_df.query(query, inplace = True)

            if len(temp_df) == 0:
                # masih belum ditemukan
                nama_model = intent_predict
                return predict_response(nama_model, sentences)
            else:
                temp_df = temp_df.reset_index(drop=True)
                # Confidence Threshold
                confidence_scores = max(intent_classifier.predict_proba(X_test_vecs)[0])
                return temp_df["text_output"][0], confidence_scores

                # return temp_df["intents"][0]

        # 2. ======================================================================================

        response, conf_score = predict_response("root", sentences)

        if conf_score > 0.3:
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
                "data": "Maaf saya belum mengerti maksud anda dan mohon menggunakan bahasa Indonesia yang baku."
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
    # app.run(debug=True)