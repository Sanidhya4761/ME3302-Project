from tensorflow.keras.models import load_model
import cv2
# Load the pre-trained model
model = load_model('model.h5')
image =cv2.imread("testData/img_0242.jpg")
image = cv2.resize(image, (256, 256))
image = tf.expand_dims(image,axis=0)
print(image.shape)
#Replace with ur custom file
k = (model.predict(image))
if(k>=0.5):
  print('No defect')
else:
  print("Defected")
