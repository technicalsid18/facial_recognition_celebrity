import cv2
import os
import numpy as np

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory where your dataset is stored
dataset_path = 'Dataset'  # Modify this to the path of your dataset

# Lists to store face images and their corresponding labels
faces = []
labels = []
label_ids = {}
current_label_id = 0

# Loop through each folder in the dataset directory
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    # Assign a unique label to each person
    if person_name not in label_ids:
        label_ids[person_name] = current_label_id
        current_label_id += 1
    label_id = label_ids[person_name]

    # Loop through each image file in the person's folder
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        # Read the image in grayscale (important for face recognition)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        # Detect faces in the image
        faces_rects = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

        # If faces are detected, add them to the training data
        for (x, y, w, h) in faces_rects:
            face_roi = image[y:y + h, x:x + w]  # Extract the face region of interest (ROI)
            faces.append(face_roi)
            labels.append(label_id)
            print(f"Added face from {image_path} with label {label_id}")

# Check if we have enough data
if len(faces) == 0:
    print("No faces found. Please check your dataset.")
else:
    print(f"Training on {len(faces)} faces with {len(set(labels))} unique labels.")

    # Train the recognizer with the collected faces and labels
    recognizer.train(faces, np.array(labels))

    # Save the trained model
    recognizer.save('trained_model.yml')
    print("Training complete. Model saved as 'trained_model.yml'.")

    # Save the label IDs mapping for later use (to recognize people by name)
    np.save('label_ids.npy', label_ids)
    print("Label IDs saved to 'label_ids.npy'.")
