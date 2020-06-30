import numpy as np
import cv2
from keras.models import load_model
from statistics import mode

# Paths to models
detection_model_path = 'trained_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'trained_models/val_acc_0.68.hdf5'
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}


frame_window = 10
x_off, y_off = 20, 20

# loading models and target size
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
target_size = emotion_classifier.input_shape[1:3]


window = []
cv2.namedWindow('Your_webcam_is_on') # I chose this name so people don't forget. :)
video_capture = cv2.VideoCapture(0)
while True:
    cam_image = video_capture.read()[1] # gets the image
    gray_img = cv2.cvtColor(cam_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
    faces = face_detection.detectMultiScale(gray_img, 1.3, 5, minSize=(48,48), maxSize=(300,300)) # detects multiple faces

    for face_coordinates in faces:
        x, y, w, h = face_coordinates
        
        x1, x2, y1, y2 = (x - x_off, x + w + x_off, y - y_off, y + h + y_off) # applies offsets
        gray_face1 = gray_img[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face1, (target_size))
        except:
            gray_face = gray_face1
            continue

        gray_face = gray_face / 255.0
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        
        #Predicting the image with the trained classifier model.
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_label_arg =  np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        
        window.append(emotion_text)
        if len(window) > frame_window:
            window.pop(0)
        try:
            emotion_mode = mode(window)
        except:
            continue

        color =  np.asarray((255,182,193)).astype(int).tolist()
        
        # Placing the text
        cv2.putText(rgb_image, emotion_mode, (x, y-45), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color, thickness=3)
        
        # Placing the rectangle
        cv2.rectangle(rgb_image, (x-20, y-20), (x + w + x_off, y + h + y_off), color, 2)

    cam_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Your_webcam_is_on', cam_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
