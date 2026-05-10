import cv2
from deepface import DeepFace
from collections import Counter
import time
import os

# =========================================
# CREATE SNAPSHOTS FOLDER
# =========================================
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# =========================================
# LOAD FACE DETECTOR
# =========================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =========================================
# OPEN WEBCAM
# =========================================
cap = cv2.VideoCapture(0)

# HD Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =========================================
# CHECK CAMERA
# =========================================
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("===================================")
print(" AI Emotion Detection Started")
print(" Press Q to Quit")
print("===================================")

# =========================================
# VARIABLES
# =========================================
emotion_history = []

emotion_counter = {}

prev_time = time.time()

# =========================================
# MAIN LOOP
# =========================================
while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # =========================================
    # MIRROR EFFECT
    # =========================================
    frame = cv2.flip(frame, 1)

    # =========================================
    # REMOVE NOISE
    # =========================================
    frame = cv2.GaussianBlur(
        frame,
        (3, 3),
        0
    )

    # =========================================
    # BRIGHTNESS IMPROVEMENT
    # =========================================
    frame = cv2.convertScaleAbs(
        frame,
        alpha=1.3,
        beta=25
    )

    # =========================================
    # GRAYSCALE
    # =========================================
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # =========================================
    # FACE DETECTION
    # =========================================
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(120, 120)
    )

    # =========================================
    # PROCESS EACH FACE
    # =========================================
    for (x, y, w, h) in faces:

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        # Crop face
        face = frame[y:y+h, x:x+w]

        try:

            # =========================================
            # AI EMOTION ANALYSIS
            # =========================================
            result = DeepFace.analyze(
                face,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )

            emotions = result[0]['emotion']

            # =========================================
            # EMOTION SCORES
            # =========================================
            happy = emotions['happy']
            sad = emotions['sad']
            angry = emotions['angry']
            fear = emotions['fear']
            surprise = emotions['surprise']
            neutral = emotions['neutral']
            disgust = emotions['disgust']

            # =========================================
            # IMPROVED SMART LOGIC
            # =========================================

            # Shock
            if surprise > 25 and fear > 15:
                emotion = "Shock"
                confidence = surprise

            # Surprise
            elif surprise > 30:
                emotion = "Surprised"
                confidence = surprise

            # Smile / Happy
            elif happy > 35:
                emotion = "Smile / Happy"
                confidence = happy

            # Angry
            elif angry > 30:
                emotion = "Angry"
                confidence = angry

            # Fear
            elif fear > 25:
                emotion = "Fear"
                confidence = fear

            # Sad
            elif sad > 35 and neutral < 40:
                emotion = "Sad"
                confidence = sad

            # Neutral
            elif neutral >= 40:
                emotion = "Neutral"
                confidence = neutral

            # Disgust
            elif disgust > 20:
                emotion = "Disgust"
                confidence = disgust

            # Default
            else:
                emotion = "Neutral"
                confidence = neutral

            # =========================================
            # STABLE EMOTION PREDICTION
            # =========================================
            emotion_history.append(emotion)

            # Keep only last 15 emotions
            if len(emotion_history) > 15:
                emotion_history.pop(0)

            # Most common emotion
            stable_emotion = Counter(
                emotion_history
            ).most_common(1)[0][0]

            # =========================================
            # EMOTION COUNTER
            # =========================================
            if stable_emotion in emotion_counter:
                emotion_counter[stable_emotion] += 1
            else:
                emotion_counter[stable_emotion] = 1

            # =========================================
            # SAVE SNAPSHOT
            # =========================================
            if confidence > 90:

                filename = (
                    f"snapshots/"
                    f"{stable_emotion}_"
                    f"{int(time.time())}.jpg"
                )

                cv2.imwrite(
                    filename,
                    frame
                )

            # =========================================
            # DISPLAY EMOTION
            # =========================================
            cv2.putText(
                frame,
                stable_emotion,
                (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3
            )

            # =========================================
            # DISPLAY CONFIDENCE
            # =========================================
            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (x, y + h + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        except Exception as e:
            print("Error:", e)

    # =========================================
    # FPS CALCULATION
    # =========================================
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # =========================================
    # PROJECT TITLE
    # =========================================
    cv2.putText(
        frame,
        "Advanced AI Emotion Detection",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        3
    )

    # =========================================
    # FPS DISPLAY
    # =========================================
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # =========================================
    # EMOTION STATISTICS
    # =========================================
    y_offset = 130

    for emo, count in emotion_counter.items():

        cv2.putText(
            frame,
            f"{emo}: {count}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        y_offset += 30

    # =========================================
    # SHOW WINDOW
    # =========================================
    cv2.imshow(
        "AI Emotion Detector",
        frame
    )

    # =========================================
    # PRESS Q TO QUIT
    # =========================================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================
# RELEASE CAMERA
# =========================================
cap.release()

# =========================================
# CLOSE WINDOWS
# =========================================
cv2.destroyAllWindows()