import json
import string
import os
from unidecode import unidecode
dir_path=os.path.dirname(os.path.realpath(__file__))
dir_path=dir_path.replace("\\","//")
def limpiar_texto(texto):
    texto = texto.lower()
    texto = unidecode(texto)
    texto = ''.join(char for char in texto if char not in string.punctuation)
    return texto

def limpiar_json(data):
    for intent in data['intents']:
        intent['patterns'] = [limpiar_texto(pattern) for pattern in intent['patterns']]
        intent['responses'] = [limpiar_texto(response) for response in intent['responses']]
    return data

# Leer el archivo JSON
with open(dir_path+'/data_bot\data_bot-main/data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Limpiar el contenido del JSON
data_limpia = limpiar_json(data)

# Guardar el JSON limpio en un nuevo archivo
with open(dir_path+'/data_bot\data_bot-main/data.json', 'w', encoding='utf-8') as file:
    json.dump(data_limpia, file, ensure_ascii=False, indent=4)

print("El archivo ha sido limpiado y guardado como 'data_limpio.JSON'")
