import cv2
import face_recognition
import os
import csv
import pickle
import numpy as np
from datetime import datetime

students = {}

with open("students.csv", "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        students[row["Name"]] = {
            "roll": row["RollNo"],
            "course": row["Course"]
        }

with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

known_face_encodings = data["encodings"]

known_face_names = []
for n in data["names"]:
    clean_name = n.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
    known_face_names.append(clean_name)

print("Training complete!")

if not os.path.exists("attendance.csv"):
    with open("attendance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "RollNo", "Course", "Date", "Time"])

marked_today = set()

today = datetime.now().strftime("%Y-%m-%d")

with open("attendance.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        if len(row) >= 5:
            if row[3] == today:
                marked_today.add(row[0])

proof_folder = "attendance_photos"
os.makedirs(proof_folder, exist_ok=True)

video_capture = cv2.VideoCapture(0)

print("Camera starting...")

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):

        name = "Unknown"
        color = (0, 0, 255)
        confidence = 0

        distances = face_recognition.face_distance(
            known_face_encodings,
            face_encoding
        )

        if len(distances) > 0:
            best_match_index = np.argmin(distances)

            if distances[best_match_index] < 0.5:
                name = known_face_names[best_match_index]

                # Safety cleanup
                name = name.replace(".jpg", "").replace(".png", "")

                color = (0, 255, 0)
                confidence = round(
                    (1 - distances[best_match_index]) * 100, 2
                )

        top, right, bottom, left = face_location
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        if name in students:
            roll = students[name]["roll"]
            course = students[name]["course"]
        else:
            roll = "-"
            course = "-"

        if name != "Unknown" and name in students:

            if name not in marked_today:

                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                with open("attendance.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, roll, course, date, time])

                marked_today.add(name)

                filename = f"{name}_{date}_{time.replace(':','-')}.jpg"
                path = os.path.join(proof_folder, filename)
                cv2.imwrite(path, frame)

                cv2.putText(frame,
                            "Attendance Marked",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            3)

            else:
                cv2.putText(frame,
                            "Already Marked Today",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            3)

        cv2.rectangle(frame,
                      (left, top),
                      (right, bottom),
                      color,
                      2)

        cv2.putText(frame,
                    f"Name: {name}",
                    (left, top - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

        cv2.putText(frame,
                    f"Roll: {roll}",
                    (left, top - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

        cv2.putText(frame,
                    f"{course}",
                    (left, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2)

        cv2.putText(frame,
                    f"Confidence: {confidence}%",
                    (left, bottom + 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2)

    cv2.imshow("Automated Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
