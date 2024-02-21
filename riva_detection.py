import cv2
import numpy as np
from keras.models import model_from_json
from riva.api import RivaAPI

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Create RivaAPI instance
riva = RivaAPI()

# start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    
    # Perform face detection using RIVA
    faces = riva.get_faces(frame)

    # Process each detected face
    for face in faces:
        # Extract face coordinates
        x, y, w, h = face["bounding_box"]

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

        # Extract the face region
        face_img = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_face, (48, 48)), -1), 0)

        # Perform emotion recognition using your emotion_model
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_label = emotion_dict[maxindex]

        # Display the predicted emotion
        cv2.putText(frame, emotion_label, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
