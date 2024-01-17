# Import necessary libraries
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, preprocessing, applications
import numpy as np
import logging

# Set up logger for error handling and logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

file_handler = logging.FileHandler('error.log')
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

TARGET_SIZE = (224, 224)  # Define target size for image preprocessing

# Preprocess the input image
def preprocess_image(img_path: str, target_size=TARGET_SIZE):
    try:
        img = preprocessing.image.load_img(img_path, target_size=target_size)
        img_array = preprocessing.image.img_to_array(img)
        if img_array.shape[2] != 3:  # Check if input image has 3 color channels
            raise ValueError("Input image does not have 3 color channels")
        return np.expand_dims(img_array / 255.0, axis=0)  # Normalize pixel values and add batch dimension
    except Exception as e:
        logger.error(f"Error occurred while processing image: {e}")
        return None

# Define VGG feature extractor model
def define_vgg_feature_extractor():
    vgg_model = applications.VGG16(weights='imagenet', include_top=True)
    return tf.keras.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc2').output)

# Load and update dataset with Flickr30k data
def load_and_update_dataset(data_csv_path: str, image_dir: str):
    try:
        flickr_data = pd.read_csv(data_csv_path, delimiter='|')  # Read data from CSV file
        image_paths = [os.path.join(image_dir, img_name) for img_name in flickr_data['image_name']]
        labels = flickr_data[' comment']  # Updated the column name to remove space
        return image_paths, labels
    except Exception as e:
        logger.error(f"Error occurred while loading and updating dataset: {e}")
        return [], []

# Extract VGG features for each image in the dataset
def extract_vgg_features(feature_extractor, image_paths):
    preprocessed_images = [preprocess_image(img_path) for img_path in image_paths if img_path is not None]
    return [feature_extractor.predict(img) for img in preprocessed_images]

# Define the RNN model for captioning
def define_captioning_model(max_length: int, vocab_size: int):
    inputs1 = layers.Input(shape=(4096,))  # Input layer for image features
    fe1 = layers.Dense(256, activation='relu')(inputs1)
    inputs2 = layers.Input(shape=(max_length,))  # Input layer for text sequences
    se1 = layers.Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = layers.LSTM(256)(se1)
    decoder1 = layers.add([fe1, se2])
    decoder2 = layers.Dense(256, activation='relu')(decoder1)
    outputs = layers.Dense(vocab_size, activation='softmax')(decoder2)
    model = models.Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

# Compile and train the RNN model with early stopping and model checkpointing
def compile_and_train_model(model, X_train, y_train, X_val, y_val, image_features, image_features_val):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile the model
    
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3)  # Use early stopping to prevent overfitting
    model_checkpoint = callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)  # Save the best model
    
    history = model.fit([image_features, X_train], y_train, validation_data=([image_features_val, X_val], y_val), epochs=10, callbacks=[early_stopping, model_checkpoint])  # Train the model
    return history

# Main function to orchestrate the workflow
def main():
    data_csv_path = "flickr30k_images/results.csv"  # Path to Flickr30k dataset CSV file
    image_dir = "flickr30k_images/flickr30k_images/flickr30k_images/"  # Directory containing the images
    
    # Load and update dataset
    image_paths, labels = load_and_update_dataset(data_csv_path, image_dir)
    
    if not image_paths:
        logger.error("No valid image paths found. Exiting.")  # Log an error message if no valid image paths are found
        return
    
    feature_extractor = define_vgg_feature_extractor()  # Define VGG feature extractor
    
    image_features = extract_vgg_features(feature_extractor, image_paths)  # Extract VGG features for the images
    
    max_length = 20  # Maximum length of caption sequence
    vocab_size = 10000  # Size of the vocabulary
    model = define_captioning_model(max_length, vocab_size)  # Define the captioning model
    
    X_train, X_val, y_train, y_val = [], [], [], []  # Initialize training and validation data
    image_features_val = []  # Placeholder for validation image features
    
    # Compile and train the model
    history = compile_and_train_model(model, X_train, y_train, X_val, y_val, image_features, image_features_val)

if __name__ == "__main__":
    main()  # Entry point of the script
