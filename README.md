# Automated Student Attendance System Using Facial Recognition

A Python-based student attendance system that uses facial recognition and OpenCV to recognize registered students through a webcam and record their attendance automatically.

## Project Overview

This project automates the attendance process using computer vision. It compares faces captured by a webcam with previously stored face encodings and records recognized students in a CSV file.

## Features

- Real-time face recognition using a webcam
- Face encoding and training from a student image dataset
- Automatic attendance recording in CSV format
- Student details stored in a CSV file
- Python-based implementation

## Technologies Used

- Python
- OpenCV
- face_recognition
- dlib
- NumPy
- CSV and Pickle

## Project Files

- `train_faces.py` — Processes student images and saves face encodings.
- `attendance_system.py` — Recognizes faces and records attendance.
- `students.csv` — Stores student information.
- `encodings.pickle` — Stores trained face encodings.

## Requirements

Python 3 and the required Python libraries are needed. A webcam and a properly organized student image dataset are also required.

Install the main dependencies using pip:

```bash
pip install opencv-python face_recognition numpy
```

The `face_recognition` package may require additional dependencies, including dlib.

## How to Run

1. Clone or download this repository.
2. Install the required dependencies.
3. Prepare the student image dataset in the expected folder structure.
4. Update `students.csv` with the appropriate student details.
5. Run the training script:

   ```bash
   python train_faces.py
   ```

6. Run the attendance system:

   ```bash
   python attendance_system.py
   ```

7. Check the generated attendance CSV file.

## Output

The system identifies registered students through the webcam and records their attendance with relevant details.

## Limitations

Recognition performance may vary with lighting, camera quality, face angle, and dataset quality. This project is intended for academic and demonstration purposes.

## Author

Amit Kumar

## License

No license has been specified for this project yet.nCV
