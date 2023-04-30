import tensorflow as tf

m = tf.keras.models.load_model('model.h5')
print(m.summary())
# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(m)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to disk
tflite_model_path = 'model1.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)