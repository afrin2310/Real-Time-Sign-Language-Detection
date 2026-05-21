import cv2
import numpy as np

_hands      = None
_mp_hands   = None
_mp_drawing = None
_mp_styles  = None

def _init():
    global _hands, _mp_hands, _mp_drawing, _mp_styles
    try:
        import mediapipe as mp
        _mp_hands   = mp.solutions.hands
        _mp_drawing = mp.solutions.drawing_utils
        _mp_styles  = mp.solutions.drawing_styles
        _hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return
    except AttributeError:
        pass
    try:
        from mediapipe.python.solutions import hands as _hm
        from mediapipe.python.solutions import drawing_utils  as _du
        from mediapipe.python.solutions import drawing_styles as _ds
        _mp_hands, _mp_drawing, _mp_styles = _hm, _du, _ds
        _hands = _hm.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception as e:
        raise RuntimeError(f"MediaPipe init failed: {e}\nRun: pip install mediapipe==0.10.9 protobuf==3.20.3")

_init()

NUM_HANDS    = 2
FEATURE_SIZE = NUM_HANDS * 21 * 3   # 126 floats total


def extract_landmarks(frame, static_mode=False):
    """
    Works for LIVE camera frames (static_mode=False)
    and for dataset images    (static_mode=True).

    Returns
    -------
    features  : np.array (126,)  — 2 hands × 21 pts × (x,y,z), zero-padded if <2 hands
    annotated : BGR frame with skeleton(s) drawn
    bboxes    : list of (xmin,ymin,xmax,ymax) per detected hand
    """
    if static_mode:
        _hands.static_image_mode          = True
        _hands.min_detection_confidence   = 0.3   # lower threshold for dataset images
    else:
        _hands.static_image_mode          = False
        _hands.min_detection_confidence   = 0.5

    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result    = _hands.process(rgb)
    annotated = frame.copy()
    bboxes    = []

    all_coords = []   # will hold up to 2 × 63-float arrays

    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            _mp_drawing.draw_landmarks(
                annotated, hand_lm,
                _mp_hands.HAND_CONNECTIONS,
                _mp_styles.get_default_hand_landmarks_style(),
                _mp_styles.get_default_hand_connections_style(),
            )

            h, w, _ = frame.shape
            xs = [lm.x for lm in hand_lm.landmark]
            ys = [lm.y for lm in hand_lm.landmark]
            xmin = max(0, int(min(xs)*w) - 20)
            xmax = min(w, int(max(xs)*w) + 20)
            ymin = max(0, int(min(ys)*h) - 20)
            ymax = min(h, int(max(ys)*h) + 20)
            bboxes.append((xmin, ymin, xmax, ymax))

            wrist  = hand_lm.landmark[0]
            coords = []
            for lm in hand_lm.landmark:
                coords.extend([lm.x - wrist.x,
                                lm.y - wrist.y,
                                lm.z - wrist.z])
            all_coords.append(coords)

    # Pad to always produce a 126-float vector (2 hands)
    while len(all_coords) < NUM_HANDS:
        all_coords.append([0.0] * 63)

    features = np.array(all_coords[:NUM_HANDS], dtype=np.float32).flatten()
    return features, annotated, bboxes


def draw_bboxes(frame, bboxes, label=None, confidence=None):
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        color = (0, 255, 150) if i == 0 else (0, 200, 255)
        cv2.rectangle(frame, (xmin-2, ymin-2), (xmax+2, ymax+2), (0,180,80), 1)
        cv2.rectangle(frame, (xmin,   ymin),   (xmax,   ymax),   color, 2)
        if i == 0 and label:
            text = f"{label}  {confidence:.0%}" if confidence else label
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (xmin, ymin-th-14), (xmin+tw+10, ymin), color, -1)
            cv2.putText(frame, text, (xmin+5, ymin-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    return frame
