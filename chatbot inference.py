# import library
import json
from string import punctuation
import random
import pickle
from tensorflow.keras.models import load_model

with open("data/intents.json") as data_file:
    data = json.load(data_file)
model = load_model('bot_model.tf')
le_filename = open("label_encoder.pickle", "rb")
le = pickle.load(le_filename)
le_filename.close()

def preprocess_string(string):
    string = string.lower()
    exclude = set(punctuation)
    string = ''.join(ch for ch in string if ch not in exclude)
    return string

def chat(model):
    print("Anda akan dihubungkan ke bot kami, mohon ditunggu")
    exit = False
    while not exit:
        inp = input("Anda : ")
        inp = preprocess_string(inp)
        prob = model.predict([inp])
        results = le.classes_[prob.argmax()]
        if prob.max() < 0.2:
            print("Bot : Maaf kak, aku ga ngerti")
        else:
            for tg in data['intents']:
                if tg['tag'] == results:
                    responses = tg['responses']
            if results == 'bye':
                exit = True
                print("END CHAT")
            print(f"Bot : {random.choice(responses)}")

if __name__ == "__main__":
    chat(model)