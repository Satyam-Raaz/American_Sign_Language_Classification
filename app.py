from collections import Counter, deque

import cv2
import numpy as np

from function import (
    actions,
    build_model,
    draw_styled_landmarks,
    get_hand_bbox,
    load_weights_h5py,
    mediapipe_detection,
    mp_hands,
    preprocess_image,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL_H5 = "model.h5"
THRESHOLD = 0.80      # minimum average confidence to accept a prediction
WINDOW = 10           # number of recent frames used for smoothing
MIN_VOTES = 7         # frames (out of WINDOW) that must agree
FLIP_FRAME = False    # set True if your predictions look mirrored/wrong
CAMERA_INDEX = 0      # or an IP camera URL, e.g. "http://192.168.1.5:8080/video"


# ---------------------------------------------------------------------------
# Load the model
# ---------------------------------------------------------------------------
def load_sign_model():
    # The architecture is rebuilt in code (function.build_model), so the saved
    # model config -- the part that breaks across Keras versions -- is never read.
    model = build_model()
    try:
        model.load_weights(MODEL_H5)
    except Exception as e:
        print(f"model.load_weights failed ({str(e).strip().splitlines()[-1][:100]})")
        print("Reading the weights directly from model.h5 with h5py...")
        load_weights_h5py(model, MODEL_H5)
    return model


model = load_sign_model()
print("Model input shape :", model.input_shape)    # expect (None, 200, 200, 3)
print("Model output shape:", model.output_shape)   # expect (None, 36)

# ---------------------------------------------------------------------------
# Real-time recognition
# ---------------------------------------------------------------------------
recent = deque(maxlen=WINDOW)   # (class_index, probabilities) for recent frames
current_text = ""
current_conf = 0.0

cap = cv2.VideoCapture(CAMERA_INDEX)

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)

        # 1) Find the hand with MediaPipe
        _, results = mediapipe_detection(frame, hands)
        bbox = get_hand_bbox(results, frame.shape)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                # 2) Classify the cropped hand with the CNN
                probs = model.predict(preprocess_image(crop), verbose=0)[0]
                recent.append((int(np.argmax(probs)), probs))

                # 3) Smooth over the last few frames to avoid flicker
                if len(recent) == WINDOW:
                    votes = Counter(idx for idx, _ in recent)
                    best_idx, count = votes.most_common(1)[0]
                    avg_conf = float(np.mean([p[best_idx] for _, p in recent]))
                    if count >= MIN_VOTES and avg_conf >= THRESHOLD:
                        current_text = str(actions[best_idx]).upper()
                        current_conf = avg_conf * 100

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_styled_landmarks(frame, results)
        else:
            recent.clear()
            current_text, current_conf = "", 0.0

        # 4) Draw the output bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (245, 117, 16), -1)
        label = f"Output: {current_text}  {current_conf:.1f}%" if current_text else "Output: -"
        cv2.putText(frame, label, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("OpenCV Feed", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()