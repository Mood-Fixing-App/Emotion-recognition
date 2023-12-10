import cv2
import numpy as np
from keras.models import load_model

# Load the emotion detection model
emotion_model_path = 'models/emotion_detection_model_2.h5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTION_COLORS = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (128, 128, 128)]
EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]

def detect_emotion(input_image_path):
    # Load the input image
    input_image = cv2.imread(input_image_path)

    # Convert the input image to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the input image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            roi = roi.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=0)

            # Predict emotion for the face
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            # Draw a rectangle around the detected face
            cv2.rectangle(input_image, (x, y), (x + w, y + h), EMOTION_COLORS[preds.argmax()], 2)

            # Add a black mask below the text for better readability
            overlay = input_image.copy()
            cv2.rectangle(overlay, (x, y - 25), (x + w, y), (0, 0, 0), -1)  # Black rectangle as the mask
            cv2.addWeighted(overlay, 1, input_image, 0.1, 0, input_image)  # Alpha blending

            # Write the emotion label on the image
            cv2.putText(input_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, EMOTION_COLORS[preds.argmax()], 1)

        # Display the input image with detected emotion
        cv2.imshow('Emotion Detection', input_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Return the detected emotion
        return label
    else:
        print("No faces detected in the input image.")
        return None

# Example usage
# input_image_path = './emotions/Scared.jpg'
# input_image_path = './emotions/Scared.jpg'
input_image_path = './emotions/Sad.jpeg'
# input_image_path = './emotions/Happy.jpeg'
detected_emotion = detect_emotion(input_image_path)
if detected_emotion:
    print("Detected Emotion:", detected_emotion)