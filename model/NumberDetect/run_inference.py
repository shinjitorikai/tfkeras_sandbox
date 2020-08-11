import pathlib
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from PIL import Image,ImageDraw

print('TensorFlow ver',tf.__version__)

# モデルの読み込み
def load_model():
    #model_path = pathlib.Path('.')/'saved_model'
    #model = tf.saved_model.load(str(model_path))
    model = tf.saved_model.load('./saved_model')
    #print(model.summary())
    model = model.signatures['serving_default']
    return model

detection_model = load_model()

#image_path = './test1.jpg'
image_path = './test3.jpg'
img = Image.open(image_path)
img_array = np.array(img)

input_tensor = tf.convert_to_tensor(img_array)
input_tensor = input_tensor[tf.newaxis,...]

output_dict = detection_model(input_tensor)

print(detection_model.output_dtypes)
print(detection_model.output_shapes)

#print(output_dict)
num_detections = output_dict['num_detections'].numpy()
print('num_detections : ',num_detections[0])
detection_classes = output_dict['detection_classes'].numpy()
print('detection_classes : ',detection_classes[0,0])
detection_scores = output_dict['detection_scores'].numpy()
print('detection_scores : ',detection_scores[0,0])
#print('detection_scores : ',detection_scores)
detection_boxes = output_dict['detection_boxes'].numpy()
print('detection_boxes : ',detection_boxes[0,0])

img_draw = ImageDraw.Draw(img)
box = detection_boxes[0,0]
img_draw.rectangle((img.width*box[1],img.height*box[0],img.width*box[3],img.height*box[2]),fill=None,outline=(255,0,0),width=5)
img.show()
