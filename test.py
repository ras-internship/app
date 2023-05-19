import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("/home/amf/ras-internship/app/backend/app/model_vvp.h5")

# Print a summary of the model architecture
model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file="best_model.png", show_shapes=True, show_layer_names=True)