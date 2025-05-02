from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('my_model.h5')

#Replace with ur custom file
model.predict('testfile.jpeg') 
model.summary()
