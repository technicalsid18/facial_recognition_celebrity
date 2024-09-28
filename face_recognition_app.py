import cv2
import numpy as np

# Load the trained model and label-to-name mapping
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.yml')
label_ids = np.load('label_ids.npy', allow_pickle=True).item()  # Load label-to-name mapping

# Invert the dictionary for quick lookup of names by label ID
label_ids_reverse = {v: k for k, v in label_ids.items()}

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
try:
# Start capturing video from webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to capture video.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Recognize each detected face
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]

            # Predict the label and confidence
            label_id, confidence = recognizer.predict(face_roi)

            # Check the confidence level and display the name
            if confidence < 70:  # You can adjust the confidence threshold
                person_name = label_ids_reverse.get(label_id, "Unknown")
                label_text = f"{person_name} ({round(100 - confidence)}%)"
            else:
                label_text = "Unknown"

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the name and confidence on the image
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Resize frame for faster processing (optional)
        frame = cv2.resize(frame, (640, 480))

        # Display the frame with the recognized faces
        cv2.imshow('Face Recognizer', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    # Your video capture code
except Exception as e:
    print(f"Error: {e}")
finally:
    video_capture.release()
    cv2.destroyAllWindows()

