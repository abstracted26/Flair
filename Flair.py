from flair.data import Sentence
from flair.models import SequenceTagger
# Importing JSON & Regex for creating a json & manipulating datastructures
import json
# Importing Flask for creating a python server
from flask import Flask, request
app = Flask(__name__)

# ENGLISH NLP Server API FOR SCREENSHOT & CHAT UPLOAD
@app.route('/nlp/', methods=['POST'])
def extract():
    print("Request Recieved")
    entities_list = []
    messages = request.form["messages"]
    # make a sentence
    sentence = Sentence(messages)

    # load the NER tagger
    tagger = SequenceTagger.load('ner-ontonotes-fast')

    # run NER over sentence
    tagger.predict(sentence)

    print(sentence)
    print('The following NER tags are found:')

    # iterate over entities and print
    for entity in sentence.get_spans('ner'):
        print(entity)
        entities_list.append(entity)

    # Creating the JSON that will be returned
    x = {
        "output": "yes"
    }
    print(x)
    return x


if __name__ == '__main__':
    from waitress import serve
    serve(app,host='0.0.0.0',port=30080)
