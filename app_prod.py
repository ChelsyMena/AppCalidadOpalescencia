from click import style
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import cv2
import os
from keras.models import load_model
from PIL import Image, ImageOps
from threading import Thread

import numpy as np
import regex as re
import glob as glob
import warnings
warnings.filterwarnings("ignore")

# Lista analistas

analistas = ['ALBEIRO ANTONIO CARMONA', 'ALBERTO DE JESUS MEJIA ARENAS', 'ALVARO MARTIN MUNERA MONSALVE', 'ANIBAL RUEDA HERRERA', 'CARLOS ARTURO RESTREPO RESTREPO',
             'DIEGO DE JESUS BUSTAMANTE MORALES', 'DIEGO HORACIO CATAÑO YEPES', 'EDYN ALVARO BERRIO TRIANA', 'ELADIO DE JESUS ECHAVARRIA AGUDELO', 'ELKIN IGNACIO MUÑOZ ALVAREZ',
             'FERNANDO VALDES PEREZ', 'FREDDY ALBERTO BEDOYA SALAZAR', 'FREDY VANEGAS GARCIA', 'GERMAN GALLEGO', 'HAROLD GALLEGO', 'HERNANDO DE JESUS QUINCHIA HENAO', 
             'IGNACIO DE JESUS ALZATE CASTRO','IVAN DARIO CARVAJAL MUÑOZ', 'JAIME HURTADO JARAMILLO', 'JAVIER ANTONIO RIOS SANCHEZ', 'JHON EDISON CASTILLO',
             'JOHN FREDY CARDONA CASTAÑO', 'JORGE ALBERTO TOBON RESTREPO','JOSE ALBERTO ROJAS QUINTERO', 'JOSÉ SANCHEZ', 'JUAN DAVID BARRENECHE USME', 'LEONARDO CANO MONROY',
             'LEONARDO LONDOÑO SALAZAR','LEONEL GIRALDO FLOREZ','LUBIN ANTONIO GARCIA TORO', 'MAURICIO DE JESUS SOTO AMAYA','NELSON ALBERTO ARANGO','NILTON FABIAN HENAO AGUIRRE',
             'OMAR ALBEIRO CHAVERRA MURILLO','PEDRO LUIS ARBOLEDA GARCIA', 'RAUL ALBERTO ECHAVARRIA CORREA','REDINTON ALEXANDER HERRERA OROZCO', 'ROBERT ARLEY ESPINOSA VILLEGAS',
             'URIEL ANTONIO MONTOYA JIMENEZ','VLADIMIR DE JESUS CORREA RAMIREZ', 'WALTER ELIUD OSORNO OSORNO','WILLIAM DANIEL ZAPATA BARRERA', 'WILMAR ANDRES ORREGO MUÑOZ',
             'WILMAR GUTIERREZ', 'CASTAÑO VALENCIA JHON JAMES','CHAVERRA LUIS EDUARDO','MENESES QUINCHIA ANDRES FELIPE','JHON JAIRO ACEVEDO', 'VILLA CORTES JUAN PABLO',
             'AGUDELO CIRO JESUS DAVID','MORENO CARDONA DAMIAN ESTEBAN','LUIS ENRIQUE ALEGRIA','CARLOS ALBERTO LARREA AGUDELO', 'ELVIS DUBAN ZAPATA VANEGAS',
              'JORGE ENRIQUE HERRERA ZAPATA','JUAN DAVID BARRENECHE USME', 'JUAN PABLO VILLA CORTES', 'JESUS DAVID AGUDELO CIRO','DAMIAN ESTEBAN MORENO CARDONA', 
              'LUIS ENRIQUE ALEGRIA','OSNAIDER RODRIGUEZ ARENAS','MARIO ALEXANDER GALLEGO' ,'JUAN CAMILO BEDOYA','JHON JAIRO GALLEGO','MARIO ALEXANDER GALLEGO',
              'YOJAN ESTEBAN HERNANDEZ TAMAYO','NELSON ARCADIO JIMENEZ OSPINA','LUIS ENRIQUE LUGO SIERRA','JANER ARLEY ZAPATA LOPEZ']
    


# CUERPO DE LA APP ---------------------------------------------------

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={'background':'F0F0F0'}, children=[
    html.Div([
        html.H1('Chequeo de Calidad: Opalescencia'),
        html.P("Bienvenidos a la aplicación de chequeo de Opalescencia.", style={'textAlign': 'center'}),
        #html.P("Por favor digite los datos solicitados, luego de clic en el botón 'Abrir Cámara'", style={'textAlign': 'center'}),
        #html.P("Cuando esté satisfecho con la posición de la muestra, presione la tecla 'Enter' para tomar la foto y cerrar la cámara.", style={'textAlign': 'center'})
    ], className='header'),

    html.Div([

        html.H3("Analista:", style = {'gridColumn': '1', 'gridRow': '1', 'textAlign': 'left'}),
        dcc.Dropdown(options=analistas, style = {'height':'25px', 'width':'350px','gridColumn': '2', 'gridRow': '1'}, id = 'analista', searchable=True), #input

        html.H3("Número de Partida:", style = {'gridColumn': '1', 'gridRow': '2'}),
        dcc.Textarea(minLength='8',maxLength='8', placeholder='10000000', id='partida', style = {'height':'25px', 'width':'100px', 'gridRow': '2', 'gridColumn': '2'}), #input
        
        html.Button('Abrir Cámara', id='abrir_camara', className = 'button', style = {'gridRow': '3', 'gridColumn': '1 / 3'}),
        html.Img(id='foto_tomada', className='pic', style = {'padding':'20px', 'borderRadius': '25px', 'gridRow': '4 / 6', 'gridColumn': '1 / 3'}),
        html.Button('Clasificar', id='clasificar', className = 'button', style = {'gridRow': '6', 'gridColumn': '1 / 3'}),     

        html.H3('Estatus de Calidad', style = {'gridRow': '1', 'gridColumn': '3'}),
        html.P(id='estatus_calidad', style = {'gridRow': '2', 'gridColumn': '3'}), 
        html.H3('Observaciones', style = {'gridRow': '3', 'gridColumn': '3'}),
        dcc.Textarea(placeholder='Agregue aqui sus Comentarios', id = 'observaciones', style = {'height':'100px', 'gridRow': '4', 'gridColumn': '3'}), #input
        html.Button('Guardar', id='guardar', className = 'button', style = {'gridRow': '5', 'gridColumn': '3'}),

        html.H2(id='mensaje_final', style={'gridRow': '7', 'gridColumn': '1 / 4'})

    ], className='main_container')
])

# FUNCIONES ----------------------------------------------------
# Función que toma la foto, la guarda en la carpeta y la muestra en la app

@app.callback(Output('foto_tomada', 'src'),
              Input('abrir_camara', 'n_clicks'),
              Input('partida', 'value'))

def tomar_foto(n_clicks, value):

    partida = str(value)
    if len(partida)!= 8:
        return "/assets/partida.png"

    if n_clicks is None:
        return "/assets/sinfoto.png"

    elif n_clicks>=1: 
        cv2.namedWindow("Presione Enter para tomar la foto")
        cv2.setWindowProperty("Presione Enter para tomar la foto", cv2.WND_PROP_TOPMOST, 1)
        vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False

        while rval:
            cv2.imshow("Presione Enter para tomar la foto", frame)
            rval, frame = vc.read()
            key = cv2.waitKey(20)

            if key == 13: # salir del ciclo con enter
                break
                
        vc.release()
        cv2.destroyWindow("Presione Enter para tomar la foto")

        filepath = f"{partida}.png"
        filepath_final = fr"assets\{partida}_0.png"
        cv2.imwrite(filepath, frame)

        if os.path.exists(filepath_final):
            
            archivos = glob.glob(fr"assets\*")
            fotos = [x for x in archivos if partida in x]
            i = int(re.search('_[0-9]*', fotos[-1])[0][1:])+1
            filepath_final = fr"assets\{partida}_{i}.png"
            os.renames(filepath, filepath_final)
        else:
            os.renames(filepath, filepath_final)

        terminacion_filepath = re.search('[0-9]*_[0-9]*.png', filepath_final)[0]

        return f"assets/{terminacion_filepath}"

# Función que corre el modelo de TF y escupe el resultado
@app.callback(Output('estatus_calidad', 'children'),
              Input('clasificar', 'n_clicks'),
              Input('partida', 'value'))

def clasificar(n_clicks, value):

    model = load_model('keras_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    if n_clicks is None:
        return "Por favor tome la foto y presione el botón 'Clasificar'"

    elif n_clicks>=1: 
        
        partida = str(value)
        archivos = glob.glob(fr"assets\*")
        fotos = [x for x in archivos if partida in x]
        filepath_final = fotos[-1]
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
              Input('analista', 'value'),
              Input('observaciones', 'value'),
              Input('guardar', 'n_clicks'))

def guardar(partida, estatus_calidad, analista, observaciones, n_clicks):

    if n_clicks is not None:
        with open('resultados.txt','a') as file:
            file.write(f"\n{partida},{estatus_calidad},{analista},{observaciones}")
        print("done")
    
        return "El resultado ha sido guardado.\n Para registrar otra partida por favor recargue la página web.\nPara salir de la pagina cierre esta ventana y la ventada del comando de windows\n¡Gracias!" 

def abrir_chrome():
    os.system("""start chrome http://127.0.0.1:8050""")        

if __name__ == '__main__':
    Thread(target = abrir_chrome).run()
    app.run_server(debug=True)
