import cv2
import pickle
import numpy as np

model = pickle.load(open("face_model.pkl","rb"))
names = pickle.load(open("names.pkl","rb"))

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:

    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face,(100,100))

        face_input = face.flatten().reshape(1,-1)/255.0

        prediction = model.predict(face_input)

        name = names[prediction[0]]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(frame,name,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

    cv2.imshow("Face Recognition",frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()