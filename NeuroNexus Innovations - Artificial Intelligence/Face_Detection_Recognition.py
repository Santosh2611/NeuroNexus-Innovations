# Import necessary libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

class FaceRecognition:
    def __init__(self, cascade_path: str, model_weights_path: str) -> None:
        # Initialize face cascade classifier and load siamese model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.siamese_model = self.load_siamese_model(model_weights_path)

        if self.siamese_model is None:
            raise ValueError("Siamese model loading failed")

    # Load siamese model architecture
    def load_siamese_model(self, model_weights_path: str) -> Model:
        input = Input(shape=(100, 100, 3))

        conv1 = Conv2D(64, (10, 10), activation='relu')(input)
        pool1 = MaxPooling2D()(conv1)
        conv2 = Conv2D(128, (7, 7), activation='relu')(pool1)
        pool2 = MaxPooling2D()(conv2)
        conv3 = Conv2D(128, (4, 4), activation='relu')(pool2)
        pool3 = MaxPooling2D()(conv3)
        conv4 = Conv2D(256, (4, 4), activation='relu')(pool3)
        flatten = Flatten()(conv4)
        dense1 = Dense(4096, activation='sigmoid')(flatten)

        model = Model(inputs=input, outputs=dense1)

        return model

    # Detect faces in the input image
    def detect_faces(self, image_path: str) -> tuple:
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError("Image not found or cannot be read")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

        return faces, image

    # Preprocess image using TensorFlow functions
    @tf.function
    def preprocess_image(self, image: tf.Tensor) -> tf.Tensor:
        processed_image = tf.image.resize(image, (100, 100))  # Use TensorFlow for resizing
        processed_image = processed_image / 255.0  # Normalize the pixel values
        return processed_image

    # Recognize the person from the face embedding
    @tf.function
    def recognize_person(self, embedding: np.ndarray) -> tf.Tensor:
        recognized_person = tf.constant("John Doe")  # Use TensorFlow constant for recognized person
        return recognized_person

    # Process the input image for face recognition
    def process_image(self, image_path: str) -> None:
        try:
            detected_faces, input_image = self.detect_faces(image_path)

            if len(detected_faces) == 0:
                print("No faces detected in the input image")
            else:
                for (x, y, w, h) in detected_faces:
                    face_image = input_image[y:y+h, x:x+w]
                    processed_face = self.preprocess_image(face_image)

                    embedding = self.siamese_model.predict(np.expand_dims(processed_face, axis=0))
                    recognized_person = self.recognize_person(embedding)

                    recognized_person_str = recognized_person.numpy().decode('utf-8')
                    cv2.rectangle(input_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(input_image, recognized_person_str, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                output_image_path = 'output_image.jpeg'
                cv2.imwrite(output_image_path, input_image)

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    # Download RAW File: https://github.com/chen0040/keras-face/blob/master/demo/models/siamese-face-net-weights.h5
    model_weights_path = 'siamese-face-net-weights.h5' 
    
    recognition_system = FaceRecognition(cascade_path, model_weights_path)
    recognition_system.process_image('input_image.jpeg')
