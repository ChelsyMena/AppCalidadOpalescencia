import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

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

partida = "1001025"
#partida = input()

cv2.imwrite(f"{partida}.png", frame)
os.rename(f"{partida}.png", f'/assets/{partida}.png')

## --------------------------------------------------------------------------------------------------------
#aqui habria que correr el modelo 

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open(f'/assets/{partida}.png')
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
prediction_max = np.argmax(prediction[0])
labels = {0: 'PASA', 1: 'NO PASA', 2: 'SIN MUESTRA'}
estatus_calidad = labels[prediction_max]
observaciones = "No estoy de acuerdo"
## --------------------------------------------------------------------------------------------------------
with open('resultados.txt','a') as file:
    file.write(f"\n{partida},{estatus_calidad},{observaciones}")