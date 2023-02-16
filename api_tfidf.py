from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/ask_chatbot", methods=['GET', 'POST'])
def print_sentences():
    if request.method == "POST":
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

        # 2. ======================================================================================

        # get vector
        import pickle

        print("LOAD TF-IDF EMBEDDING")
        tf_idf = pickle.load(open("word_embedding/tf_idf.pkl", 'rb'))

        X_test_vecs = tf_idf.transform([sentences]).todense().tolist()

        # 3. ======================================================================================

        print("CLASSIFICATION")

        import pickle
        intent_classifier = pickle.load(open("intent_classification_model/svm_tf_idf_StemSastrawi_ratio8020_woSN.pkl", 'rb'))
        intent_predict = intent_classifier.predict(X_test_vecs)
        intent_predict = intent_predict[0]

        # Confidence Threshold
        import numpy as np
        confidence_scores = np.max(intent_classifier.decision_function(X_test_vecs), axis=1)
        print("confidence_scores = ", confidence_scores)

        # 4. ======================================================================================

        import pandas as pd
        df = pd.read_pickle("dataset/list_jawaban.pkl")

        query = "intents == '"+intent_predict+"'"
        df.query(query, inplace = True)

        df = df.reset_index(drop=True)
        print(df["text_output"][0])

        json_data = {
            "success": 1,
            "message": "Get response successfully",
            "data": df["text_output"][0]
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