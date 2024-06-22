#permite el procesamiento de lenguaje natural
import nltk
nltk.download("punkt")
#minimizar las palabra
from nltk.stem.lancaster import LancasterStemmer
#instanaciamos el minimizador
stemmer=LancasterStemmer()
#permite tabajar con arreglos y realizar manipulaciones conversiones etc.
import numpy
#herramienta de deep learning
import tflearn
import tensorflow
#permite manipular contenido json.
import json
#permite crear numeros aleaotorios
import random
#permite guardar los modelos de entrenamiento (mejora la velocidad, ya que no hay que entrenar desde 0 varias veces)
import pickle
import os
dir_path=os.path.dirname(os.path.realpath(__file__))
dir_path=dir_path.replace("\\","//")
with open(dir_path+'/data_bot\data_bot-main/data.json','r') as file:
    database=json.load(file)
words=[]
all_words=[]
tags=[]
aux=[]
auxA=[]
auxB=[]
training=[]
exit=[]

try:
    with open("Entrenamiento/brain.pickle","rb") as pickleBrain:
        all_words,tags,training,exit=pickle.load(pickleBrain)
    #print("ya esta creado el brain.pickle")
except:    
    for intent in database["intents"]:
        for pattern in intent["patterns"]:
            #separamos la frase en palabras
            auxWords=nltk.word_tokenize(pattern)
            #print(auxWords)
            #print("***************************************************************************")
            #guardamos las palabras
            auxA.append(auxWords)
            auxB.append(auxWords)
            #guardar los tags
            aux.append(intent["tag"])
    #simbolos a ignorar
    ignore_words=['?','!','.',',','¿',"'",'"','$','-',':','_','&','%','/',"(",")","=","*","#"]
    for w in auxB:
        if w not in ignore_words:
            words.append(w)
    import itertools
    words=sorted(set(list(itertools.chain.from_iterable(words))))
    #words=sorted(set(words))
    # print(words)
    tags=sorted(set(aux))
    #print(tags)

    #convertir a minuscula
    all_words=[stemmer.stem(w.lower()) for w in words]
    #print(all_words)
    #print(len(all_words))
    all_words=sorted(list(set(all_words)))
    #ordenar tags
    tags=sorted(tags)
    training=[]
    exit=[]
    #creamos una salida falsa
    null_exit=[0 for _ in range(len(tags))]
    #print(null_exit)
    for i,document in enumerate(auxA):
        bucket=[]
        #minuscula y quitar signos
        auxWords=[stemmer.stem(w.lower()) for w in document if w!="?"]
        #print(auxWords)
        "recorremos"
        for w in all_words:
            if w in auxWords:
                bucket.append(1)
            else:
                bucket.append(0)
        exit_row=null_exit[:]
        exit_row[tags.index(aux[i])]=1
        training.append(bucket)
        exit.append(exit_row)

    #print(training)
    #print(exit)   
    training=numpy.array(training)
    #print(training)  
    #exit=numpy.array(exit)

    #crear el archive pickle
    with open("Entrenamiento/brain.pickle","wb") as pickleBrain:
        pickle.dump((all_words,tags,training,exit),pickleBrain)

tensorflow.compat.v1.reset_default_graph()
tflearn.init_graph(num_cores=8,gpu_memory_fraction=0.9)
#creamos la red neuronal
net=tflearn.input_data(shape=[None,len(training[0])])
#redes intermedias
net=tflearn.fully_connected(net,100,activation='Relu')
net=tflearn.fully_connected(net,50)
net=tflearn.dropout(net,0.5)
# neurona salidah
net=tflearn.fully_connected(net,len(exit[0]),activation='softmax')
#red completada
net=tflearn.regression(net,optimizer='adam',learning_rate=0.01,loss='categorical_crossentropy')
model=tflearn.DNN(net)

if os.path.isfile(dir_path+"/Entrenamiento/model.tflearn.index"):
    model.load(dir_path+"/Entrenamiento/model.tflearn")
    #print("ya exise data entrenada")
else:
    model.fit(training,exit,validation_set=0.1,show_metric=True,batch_size=128,n_epoch=2000)
    model.save("Entrenamiento/model.tflearn")

import os
import time  # Importar el módulo time para medir el tiempo

# Tu código hasta el punto donde quieres limpiar la consola y comenzar la medición del tiempo
os.system('cls')  # Limpiar la consola en Windows
print("Asistente: HOLA YO SOY BAYMAX, TU ASISTENTE PERSONAL!!")

# Función para la respuesta del asistente
def response(texto):
    start_time = time.time()  # Obtener el tiempo actual al inicio de la respuesta
    
    if texto == "gracias":
        answer = "Descansa mañana será un gran día"
        return False, 0  # Retornar False indicando fin de la conversación y 0 como tiempo
    
    else:
        bucket = [0 for _ in range(len(all_words))]
        processed_sentence = nltk.word_tokenize(texto)
        processed_sentence = [stemmer.stem(palabra.lower()) for palabra in processed_sentence]
        
        for individual_word in processed_sentence:
            for i, palabra in enumerate(all_words):
                if palabra == individual_word:
                    bucket[i] = 1
        
        results = model.predict([numpy.array(bucket)])
        index_results = numpy.argmax(results)
        max_value = results[0][index_results]
        target = tags[index_results]
        
        for tagAux in database["intents"]:
            if tagAux['tag'] == target:
                answer = random.choice(tagAux['responses'])
        
        print("Asistente: " + answer)
        end_time = time.time()  # Obtener el tiempo actual al final de la respuesta
        return True, (end_time - start_time) * 1000  # Retornar True indicando que la conversación continúa y el tiempo en milisegundos

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