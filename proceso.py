import json
import registro
import main
import traducir
print("*******************************************************")
print("HOLA, YO SOY BAYMAX, TU ASISTENTE MEDICO PERSONAL!!")
print("*******************************************************")
rspt=input("¿YA ESTA REGISTRADO?\n")
print("*******************************************************")
if(rspt=="no"):
    print("PORFAVOR PERMITEME TOMAR SUS DATOS")
    print("*******************************************************")
    datos=registro.registrar()
    x = {
        "Nombre" : datos[0],
        "Edad" : datos[1],
        "Peso" : datos[2],
        "Talla" : datos[3],
        "Genero" : datos[4],
        "Enfermedades" : datos[5],
    }
    f=open("usuario.json","r")
    c=f.read()
    js=json.loads(c)
    js[datos[0]]=x
    s=json.dumps(js,indent=4)
    f=open("usuario.json","w")
    f.write(s)
    f.close()
with open("usuario.json", "r") as f:
    datos_usuarios = json.load(f)
    #print (datos_usuarios)
nom = input("¿CUÁL ES SU USUARIO (nombre)?\n")

if nom in datos_usuarios:
    usuario = datos_usuarios[nom]['Nombre']
    print("*******************************************************")
    print("*******************************************************")
    print("*******************************************************")
    print("HOLA  " + usuario + " , YO SOY BAYMAX, TU ASISTENTE MEDICO PERSONAL!!")
    bool=True
    while bool==True:
        texto=input()
        texto=traducir.correction(texto)
        bool=main.response(texto)
    print("FUE UN GUSTO AYUDARTE "+ usuario+ " !!")
else:
    print("Usuario no encontrado.")
