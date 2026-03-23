import cv2
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

name = input("Enter person name: ")
person_path = os.path.join(dataset_path, name)
os.makedirs(person_path, exist_ok=True)

camera = cv2.VideoCapture(0)

count = 0

print("Capturing faces... Press ESC to stop")

while count < 40:

    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face,(100,100))

        img_path = os.path.join(person_path,f"{count}.jpg")
        cv2.imwrite(img_path,face)

        count += 1

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Capturing Faces",frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()

print("Face images captured")

# ------------------------------
# TRAIN MODEL
# ------------------------------

faces = []
labels = []
names = []

label_id = 0

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    names.append(person)

    for img_name in os.listdir(person_path):

        img_path = os.path.join(person_path,img_name)

        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(100,100))

        faces.append(img.flatten())
        labels.append(label_id)

    label_id += 1

faces = np.array(faces)/255.0
labels = np.array(labels)

print("Training ANN model...")

model = MLPClassifier(
    hidden_layer_sizes=(512,256),
    activation="relu",
    solver="adam",
    max_iter=400
)

model.fit(faces,labels)

pickle.dump(model,open("face_model.pkl","wb"))
pickle.dump(names,open("names.pkl","wb"))

print("Model trained and saved successfully")