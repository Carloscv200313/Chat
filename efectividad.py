import os
import time
import random
import json
import numpy as np
import tflearn
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle

# Descargar datos necesarios de NLTK
nltk.download("punkt")

# Inicialización del stemmer
stemmer = LancasterStemmer()

# Cargar base de datos desde JSON
dir_path = os.path.dirname(os.path.realpath(__file__))
database_path = os.path.join(dir_path, 'data_bot', 'data_bot-main', 'data.json')

with open(database_path, 'r') as file:
    database = json.load(file)

# Inicialización de listas y variables
words = []
all_words = []
tags = []
auxA = []
training = []
exit = []

# Función para procesar patrones y generar datos de entrenamiento
def process_patterns(database):
    for intent in database["intents"]:
        for pattern in intent["patterns"]:
            # Tokenización y stemming de las palabras
            aux_words = nltk.word_tokenize(pattern)
            auxA.append(aux_words)
            tags.append(intent["tag"])

    # Filtrar palabras ignoradas y realizar stemming
    ignore_words = ['?', '!', '.', ',', '¿', "'", '"', '$', '-', ':', '_', '&', '%', '/', "(", ")", "=", "*", "#"]
    words = [w for w_list in auxA for w in w_list if w not in ignore_words]
    words = sorted(set(words))
    tags = sorted(set(tags))

    # Convertir palabras a minúsculas y aplicar stemming
    all_words = [stemmer.stem(w.lower()) for w in words]
    all_words = sorted(set(all_words))

    # Preparar datos de entrenamiento y salida esperada
    for i, document in enumerate(auxA):
        bucket = []
        aux_words = [stemmer.stem(w.lower()) for w in document if w != "?"]

        for w in all_words:
            bucket.append(1) if w in aux_words else bucket.append(0)

        exit_row = [0 for _ in range(len(tags))]
        exit_row[tags.index(tags[i])] = 1
        training.append(bucket)
        exit.append(exit_row)

    return all_words, tags, np.array(training), np.array(exit)

# Cargar datos de entrenamiento desde pickle si está disponible, sino procesar y guardar
pickle_path = os.path.join(dir_path, "Entrenamiento", "brain.pickle")
try:
    with open(pickle_path, "rb") as pickleBrain:
        all_words, tags, training, exit = pickle.load(pickleBrain)
except FileNotFoundError:
    all_words, tags, training, exit = process_patterns(database)
    with open(pickle_path, "wb") as pickleBrain:
        pickle.dump((all_words, tags, training, exit), pickleBrain)

# Reiniciar grafo de TensorFlow y definir la red neuronal con tflearn
tf.compat.v1.reset_default_graph()
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.9)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 100, activation='relu')
net = tflearn.fully_connected(net, 50)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, len(exit[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')
model = tflearn.DNN(net)

# Cargar modelo entrenado si existe, sino entrenar y guardar
model_path = os.path.join(dir_path, "Entrenamiento", "model.tflearn")
if os.path.isfile(model_path + ".index"):
    model.load(model_path)
else:
    model.fit(training, exit, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=2000)
    model.save(model_path)


# Función para la respuesta del asistente
def response(texto):
    start_time = time.time()  # Obtener el tiempo actual al inicio de la respuesta
    
    if texto.lower() == "gracias":
        answer = "Descansa, mañana será un gran día."
        return False, 0  # Retornar False indicando fin de la conversación y 0 como tiempo
    
    else:
        bucket = [0 for _ in range(len(all_words))]
        processed_sentence = nltk.word_tokenize(texto)
        processed_sentence = [stemmer.stem(palabra.lower()) for palabra in processed_sentence]
        
        for i, individual_word in enumerate(all_words):
            if individual_word in processed_sentence:
                bucket[i] = 1
        
        results = model.predict([np.array(bucket)])
        index_results = np.argmax(results)
        target = tags[index_results]
        
        for tagAux in database["intents"]:
            if tagAux['tag'] == target:
                answer = random.choice(tagAux['responses'])
                break
        
        print("Asistente: " + answer)
        end_time = time.time()  # Obtener el tiempo actual al final de la respuesta
        return True, (end_time - start_time) * 1000  # Retornar True indicando que la conversación continúa y el tiempo en milisegundos

# Evaluación del modelo en los datos de entrenamiento
predictions = model.predict(training)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(exit, axis=1)

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
# Limpiar consola y empezar interacción con usuario
os.system('cls' if os.name == 'nt' else 'clear')  # Limpiar la consola
print(f'Exactitud: {accuracy:.4f}')
print(f'Precisión: {precision:.4f}')
print(f'Recordar: {recall:.4f}')
print(f'Puntuación F1: {f1:.4f}')
print("*************************************************")
print("Asistente: HOLA YO SOY BAYMAX, TU ASISTENTE PERSONAL!!")
# Bucle principal de interacción con el usuario
conversation_time = 0
while True:
    texto = input("Usuario: ")
    continue_conversation, response_time = response(texto)
    if not continue_conversation:
        break
    conversation_time += response_time
    print(f"Tiempo de respuesta: {response_time:.2f} ms")

print(f"Tiempo total de conversación: {conversation_time:.2f} ms")
