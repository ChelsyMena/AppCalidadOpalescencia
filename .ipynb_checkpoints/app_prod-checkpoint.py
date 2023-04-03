from click import style
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import cv2
import os
import warnings
warnings.filterwarnings("ignore")
from threading import Timer

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# CUERPO DE LA APP ---------------------------------------------------

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={'background':'F0F0F0'}, children=[
    html.Div([
        html.H1('Chequeo de Calidad: Opalescencia'),
        html.P("Bienvenidos a la aplicación de chequeo de Opalescencia.", style={'textAlign': 'center'}),
        html.P("Por favor ponga la muestra encima de la cámara y presione Abrir Cámara.", style={'textAlign': 'center'}),
        html.P("Cuando esté satisfecho con la posición de la muestra, presione Enter.", style={'textAlign': 'center'})

    ], className='header'),

    html.Div([
        html.H3("Número de Partida:", style = {'gridColumn': '1', 'gridRow': '1'}),
        dcc.Textarea(minLength='8',maxLength='8', placeholder='10000000', id='partida', style = {'height':'20px', 'width':'100px', 'gridRow': '1', 'gridColumn': '2'}), #input
        html.Button('Abrir Cámara', id='abrir_camara', className = 'button', style = {'gridRow': '2', 'gridColumn': '1 / 3'}),
        html.Img(id='foto_tomada', className='pic', style = {'padding':'20px', 'borderRadius': '25px', 'gridRow': '3 / 5', 'gridColumn': '1 / 3'}),
        html.Button('Clasificar', id='clasificar', className = 'button', style = {'gridRow': '5', 'gridColumn': '1 / 3'}),     

        html.H3('Estatus de Calidad', style = {'gridRow': '1', 'gridColumn': '3'}),
        html.P(id='estatus_calidad', style = {'gridRow': '2', 'gridColumn': '3'}), 
        html.H3('Observaciones', style = {'gridRow': '3', 'gridColumn': '3'}),
        dcc.Textarea(placeholder='Agregue aqui sus Comentarios', id = 'observaciones', style = {'height':'100px', 'gridRow': '4', 'gridColumn': '3'}), #input
        html.Button('Guardar', id='guardar', className = 'button', style = {'gridRow': '5', 'gridColumn': '3'}),

        html.H2(id='mensaje_final', style={'gridRow': '6', 'gridColumn': '1 / 4'})

    ], className='main_container')
])

# FUNCIONES ----------------------------------------------------
# Función que toma la foto, la guarda en la carpeta y la muestra en la app

@app.callback(Output('foto_tomada', 'src'),
              Input('abrir_camara', 'n_clicks'),
              Input('partida', 'value'))

def tomar_foto(n_clicks, value):

    if n_clicks is None:
        return "/assets/sinfoto.png"

    elif n_clicks>=1: 
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False

        while rval:
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            key = cv2.waitKey(20)

            if key == 13: # salir del ciclo con enter
                break
                
        vc.release()
        cv2.destroyWindow("preview")

        partida = str(value)
        filepath = f"{partida}.png"
        filepath_final = fr"D:\Users\chelsy.mena\OneDrive - Centro de Servicios Mundial SAS\Documentos\En Proceso\Calidad\ML Visual\App Dash\assets\{filepath}"
        cv2.imwrite(filepath, frame)

        os.renames(filepath, filepath_final)

        return f"assets/{partida}.png"

# Función que corre el modelo de TF y escupe el resultado
@app.callback(Output('estatus_calidad', 'children'),
              Input('clasificar', 'n_clicks'),
              Input('partida', 'value'))

def clasificar(n_clicks, value):

    model = load_model('keras_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    if n_clicks is None:
        return "Por favor tome la foto y presione el botón Clasificar"

    elif n_clicks>=1: 
        
        partida = str(value)
        filepath_final = fr"D:\Users\chelsy.mena\OneDrive - Centro de Servicios Mundial SAS\Documentos\En Proceso\Calidad\ML Visual\App Dash\assets\{partida}.png"
        image = Image.open(filepath_final)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        prediction_max = np.argmax(prediction[0])
        labels = {0: 'PASA', 1: 'NO PASA', 2: 'SIN MUESTRA'}
        resultado = labels[prediction_max]

        return resultado

# Función que Guarda en el .txt
@app.callback(Output('mensaje_final', 'children'),
              Input('partida', 'value'),
              Input('estatus_calidad', 'children'),
              Input('observaciones', 'value'),
              Input('guardar', 'n_clicks'))

def guardar(partida, estatus_calidad, observaciones, n_clicks):

    if n_clicks is not None:
        with open('resultados.txt','a') as file:
            file.write(f"\n{partida},{estatus_calidad},{observaciones}")
        print("done")
    
        return "El resultado ha sido guardado.\n Para registrar otra partida por favor recargue la página web.\nPara salir de la pagina cierre esta ventana y la ventada del comando de windows\n¡Gracias!" 

def abrir_chrome:
   os.system(""":Test
                 Curl http://127.0.0.1:8050
                 If "%errorlevel%" neq "0" goto :Test
                 start chrome http://127.0.0.1:8050""")

if __name__ == '__main__':
    #Timer(2, abrir_chrome).start()
    app.run_server(debug=True)

