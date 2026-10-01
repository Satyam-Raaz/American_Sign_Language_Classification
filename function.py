import cv2
import numpy as np
import mediapipe as mp

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# ---------------------------------------------------------------------------
# Model settings (must match the training notebook)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 200  # image_size used in training (200 x 200)

# Class order used in the notebook: 0-9 then a-z (36 classes).
# flow_from_directory sorts folder names alphabetically, which gives this same order.
actions = np.array(list("0123456789abcdefghijklmnopqrstuvwxyz"))


def mediapipe_detection(image, model):
    """Run MediaPipe Hands on a BGR frame. Returns (BGR image, results)."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    """Draw hand landmarks on the image (optional, for visual feedback)."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )


def get_hand_bbox(results, frame_shape, padding=0.3):
    """
    Return a square bounding box (x1, y1, x2, y2) around the first detected
    hand, or None if no hand is found. `padding` adds a margin around the hand.
    """
    if not results.multi_hand_landmarks:
        return None

    h, w = frame_shape[:2]
    landmarks = results.multi_hand_landmarks[0].landmark
    xs = np.array([lm.x for lm in landmarks]) * w
    ys = np.array([lm.y for lm in landmarks]) * h

    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * padding)
    side = min(side, h, w)  # a square can never be bigger than the frame

    # Shift the square back inside the frame instead of cutting it off
    x1 = int(round(min(max(cx - side / 2, 0), w - side)))
    y1 = int(round(min(max(cy - side / 2, 0), h - side)))
    x2, y2 = int(x1 + side), int(y1 + side)
    return x1, y1, x2, y2


def preprocess_image(img_bgr):
    """
    Turn a BGR crop into the model input, exactly like training:
    resize to 200x200, BGR -> RGB, scale to 0-1, add batch dimension.
    Returns an array of shape (1, 200, 200, 3).
    """
    img = cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def build_model():
    """
    Rebuild the exact CNN from the training notebook (200x200x3 -> 36 classes).
    Used to load the weights from model.h5 even when the Keras version that
    saved the file is newer than the one installed here.
    """
    from keras.models import Sequential
    from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

    model = Sequential([
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

        Conv2D(32, 3, activation="relu", padding="same"),
        Conv2D(32, 3, activation="relu", padding="same"),
        MaxPooling2D(padding="same"),
        Dropout(0.2),

        Conv2D(64, 3, activation="relu", padding="same"),
        Conv2D(64, 3, activation="relu", padding="same"),
        MaxPooling2D(padding="same"),
        Dropout(0.3),

        Conv2D(128, 3, activation="relu", padding="same"),
        Conv2D(128, 3, activation="relu", padding="same"),
        MaxPooling2D(padding="same"),
        Dropout(0.4),

        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.2),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(len(actions), activation="softmax"),
    ])
    return model


def load_weights_h5py(model, path):
    """
    Last-resort loader: read the weights straight out of model.h5 with h5py and
    copy them into `model` layer by layer (in order). It never touches the saved
    model config, so it works no matter which Keras version wrote the file.
    """
    import h5py

    with h5py.File(path, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f
        names = [n.decode() if isinstance(n, bytes) else n for n in g.attrs["layer_names"]]

        saved = []  # weights per saved layer that has any (Conv2D / Dense)
        for name in names:
            found = {}
            g[name].visititems(
                lambda n, o: found.__setitem__(n.split("/")[-1], o[()])
                if isinstance(o, h5py.Dataset) else None
            )
            if found:
                saved.append((name, [found["kernel"], found["bias"]]))

        targets = [layer for layer in model.layers if layer.weights]
        if len(saved) != len(targets):
            raise ValueError(
                f"model.h5 has {len(saved)} weighted layers but the model expects {len(targets)}"
            )
        for layer, (_, weights) in zip(targets, saved):
            layer.set_weights(weights)