import cv2
from deepface import DeepFace
from collections import Counter

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

# Set HD resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("AI Emotion Detection Started")
print("Press Q to Quit")

# Store recent emotions for stable prediction
emotion_history = []

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Flip frame horizontally
    frame = cv2.flip(frame, 1)

    # Improve brightness
    frame = cv2.convertScaleAbs(
        frame,
        alpha=1.2,
        beta=30
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120)
    )

    # Loop through faces
    for (x, y, w, h) in faces:

        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        # Crop face region
        face = frame[y:y + h, x:x + w]

        try:

            # Analyze emotions
            result = DeepFace.analyze(
                face,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )

            emotions = result[0]['emotion']

            # Extract scores
            happy_score = emotions['happy']
            neutral_score = emotions['neutral']
            surprise_score = emotions['surprise']

            # Better happy/smile detection
            if happy_score > 25:
                emotion = "Smile / Happy"
                confidence = happy_score

            elif surprise_score > 20:
                emotion = "Surprised"
                confidence = surprise_score

            elif neutral_score > 40:
                emotion = "Neutral"
                confidence = neutral_score

            else:
                dominant = max(
                    emotions,
                    key=emotions.get
                )

                confidence = emotions[dominant]

                emotion_map = {
                    "happy": "Happy",
                    "sad": "Sad",
                    "angry": "Angry",
                    "fear": "Fear",
                    "surprise": "Surprised",
                    "neutral": "Neutral",
                    "disgust": "Disgust"
                }

                emotion = emotion_map.get(
                    dominant,
                    "Detecting"
                )

            # Store emotions for stable output
            emotion_history.append(emotion)

            # Keep only last 10 emotions
            if len(emotion_history) > 10:
                emotion_history.pop(0)

            # Most common emotion
            stable_emotion = Counter(
                emotion_history
            ).most_common(1)[0][0]

            # Show emotion text
            cv2.putText(
                frame,
                stable_emotion,
                (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            # Show confidence
            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (x, y + h + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        except Exception as e:
            print("Error:", e)

    # Title
    cv2.putText(
        frame,
        "AI Emotion Detection System",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        3
    )

    # Show webcam feed
    cv2.imshow(
        "Emotion Detector",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()