import cv2
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json

# -------------------- Load Model --------------------
with open("facialemotionmodel.json", "r") as f:
    model_json = f.read()

# Register Sequential explicitly (needed for TF >= 2.13)
model = model_from_json(model_json, custom_objects={"Sequential": Sequential})

# Load weights
model.load_weights("facialemotionmodel.h5")

# -------------------- Load Haar Cascade --------------------
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# -------------------- Preprocess Function --------------------
def extract_features(image):
    image = cv2.resize(image, (48, 48))       # resize to 48x48
    image = image.astype("float32") / 255.0   # normalize
    image = np.expand_dims(image, axis=0)     # add batch dimension
    image = np.expand_dims(image, axis=-1)    # add channel dimension
    return image

# -------------------- Labels --------------------
labels = {
    0: "angry", 1: "disgust", 2: "fear",
    3: "happy", 4: "neutral", 5: "sad", 6: "surprise"
}

# -------------------- Start Webcam --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing index in VideoCapture(0).")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        # Predict
        img = extract_features(roi_gray)
        preds = model.predict(img, verbose=0)[0]
        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id])
        label_text = f"{labels[class_id]}: {confidence*100:.1f}%"

        # Draw rectangle + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        print(f"Detected emotion: {label_text}")

    # Show video window
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Wait for any key press to exit
    if cv2.waitKey(1) != -1:
        print("Key pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

