import numpy as np
from PIL import Image
from tensorflow import keras
# from keras.preprocessing import image

labels = ['bacterial_leaf_blight',
          'bacterial_leaf_streak',
          'bacterial_panicle_blight',
          'blast',
          'brown_spot',
          'dead_heart',
          'downy_mildew',
          'hispa',
          'normal',
          'tungro']

model = keras.models.load_model('static\penyakit_padi_detection.h5')

def predict(img_url):
    img = Image.open(img_url)
    x = img.resize((256, 256), Image.ANTIALIAS)
    i = np.asarray(x).astype(np.float32)
    #plt.imshow(i/255.)
    x = np.expand_dims(i, axis=0)
    images = np.vstack([x])
    predictions = model.predict(images)
    l = np.argmax(predictions)
    #predictions
    return labels[l]
