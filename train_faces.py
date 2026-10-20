import face_recognition
import os
import pickle

dataset_path = "dataset"

known_face_encodings = []
known_face_names = []

for student_name in os.listdir(dataset_path):

    student_folder = os.path.join(dataset_path, student_name)

    if os.path.isdir(student_folder):

        for file in os.listdir(student_folder):

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                image_path = os.path.join(student_folder, file)

                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)

                    if len(encodings) > 0:
                        known_face_encodings.append(encodings[0])

                        known_face_names.append(student_name)

                except:
                    pass

data = {
    "encodings": known_face_encodings,
    "names": known_face_names
}

with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print("Training complete!")
