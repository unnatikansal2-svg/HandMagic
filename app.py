import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# ── Colours & UI ─────────────────────────────────────────────────────────────
COLORS = {
    "Red":    (0,   0,   255),
    "Green":  (0,   255, 0),
    "Blue":   (255, 0,   0),
    "Yellow": (0,   255, 255),
    "White":  (255, 255, 255),
}
COLOR_NAMES = list(COLORS.keys())

BRUSH_SIZE   = 8
ERASER_SIZE  = 40
HEADER_H     = 80          # height of the top toolbar
BTN_W        = 100         # button width
BTN_MARGIN   = 10

current_color_idx = 0
eraser_mode       = False
prev_x, prev_y    = None, None

# ── Canvas (persists between frames) ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot open camera.")
H, W = frame.shape[:2]
canvas = np.zeros((H, W, 3), dtype=np.uint8)   # black canvas

# ── Helper: draw toolbar ─────────────────────────────────────────────────────
def draw_toolbar(frame, cur_color_idx, eraser_on):
    # Dark toolbar background
    cv2.rectangle(frame, (0, 0), (W, HEADER_H), (30, 30, 30), -1)

    x = BTN_MARGIN
    # Colour buttons
    for i, name in enumerate(COLOR_NAMES):
        c = COLORS[name]
        border = 3 if i == cur_color_idx and not eraser_on else 1
        cv2.rectangle(frame, (x, 10), (x + BTN_W, HEADER_H - 10), c, -1)
        cv2.rectangle(frame, (x, 10), (x + BTN_W, HEADER_H - 10),
                      (255, 255, 255), border)
        cv2.putText(frame, name, (x + 8, HEADER_H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        x += BTN_W + BTN_MARGIN

    # Eraser button
    e_color = (200, 200, 200) if eraser_on else (80, 80, 80)
    cv2.rectangle(frame, (x, 10), (x + BTN_W, HEADER_H - 10), e_color, -1)
    cv2.rectangle(frame, (x, 10), (x + BTN_W, HEADER_H - 10),
                  (255, 255, 255), 3 if eraser_on else 1)
    cv2.putText(frame, "Eraser", (x + 12, HEADER_H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    x += BTN_W + BTN_MARGIN

    # Clear button
    cv2.rectangle(frame, (x, 10), (x + BTN_W, HEADER_H - 10), (0, 0, 180), -1)
    cv2.rectangle(frame, (x, 10), (x + BTN_W, HEADER_H - 10), (255, 255, 255), 1)
    cv2.putText(frame, "Clear", (x + 18, HEADER_H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return frame

# ── Helper: check toolbar click ───────────────────────────────────────────────
def check_toolbar_click(fx, fy):
    """Returns ('color', idx) | ('eraser',) | ('clear',) | None"""
    if fy > HEADER_H:
        return None
    x = BTN_MARGIN
    for i in range(len(COLOR_NAMES)):
        if x <= fx <= x + BTN_W:
            return ("color", i)
        x += BTN_W + BTN_MARGIN
    if x <= fx <= x + BTN_W:        # eraser
        return ("eraser",)
    x += BTN_W + BTN_MARGIN
    if x <= fx <= x + BTN_W:        # clear
        return ("clear",)
    return None

# ── Helper: is index finger up, others down? ──────────────────────────────────
def index_up_only(lm):
    """Returns True if only index finger is raised (drawing mode)."""
    # Tip ids: thumb=4, index=8, middle=12, ring=16, pinky=20
    index_up  = lm[8].y  < lm[6].y
    middle_dn = lm[12].y > lm[10].y
    ring_dn   = lm[16].y > lm[14].y
    pinky_dn  = lm[20].y > lm[18].y
    return index_up and middle_dn and ring_dn and pinky_dn

def two_fingers_up(lm):
    """Returns True if index + middle are raised (hover / pause mode)."""
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_dn   = lm[16].y > lm[14].y
    pinky_dn  = lm[20].y > lm[18].y
    return index_up and middle_up and ring_dn and pinky_dn

# ── Main loop ─────────────────────────────────────────────────────────────────
print("Air Canvas running — press Q to quit.")
print("Gestures:  ☝️  one finger = draw  |  ✌️  two fingers = pause/hover")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)          # mirror
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)

    if res.multi_hand_landmarks:
        lm_list = res.multi_hand_landmarks[0].landmark

        # Fingertip pixel coords (index finger tip = landmark 8)
        fx = int(lm_list[8].x * W)
        fy = int(lm_list[8].y * H)

        draw_mode  = index_up_only(lm_list)
        hover_mode = two_fingers_up(lm_list)

        # ── Toolbar interaction ───────────────────────────────────────────
        if fy < HEADER_H and draw_mode:
            action = check_toolbar_click(fx, fy)
            if action:
                if action[0] == "color":
                    current_color_idx = action[1]
                    eraser_mode = False
                elif action[0] == "eraser":
                    eraser_mode = not eraser_mode
                elif action[0] == "clear":
                    canvas[:] = 0
            prev_x, prev_y = None, None   # don't draw while clicking toolbar

        # ── Drawing on canvas ─────────────────────────────────────────────
        elif draw_mode and fy >= HEADER_H:
            if prev_x is not None and prev_y is not None:
                if eraser_mode:
                    cv2.line(canvas, (prev_x, prev_y), (fx, fy),
                             (0, 0, 0), ERASER_SIZE)
                else:
                    color = COLORS[COLOR_NAMES[current_color_idx]]
                    cv2.line(canvas, (prev_x, prev_y), (fx, fy),
                             color, BRUSH_SIZE)
            prev_x, prev_y = fx, fy

        else:
            prev_x, prev_y = None, None   # lift pen when gesture changes

        # Draw skeleton
        mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0],
                               mp_hands.HAND_CONNECTIONS)

        # Draw cursor circle
        cur_color = (150, 150, 150) if eraser_mode else COLORS[COLOR_NAMES[current_color_idx]]
        r = ERASER_SIZE // 2 if eraser_mode else BRUSH_SIZE
        cv2.circle(frame, (fx, fy), r, cur_color, 2)

    else:
        prev_x, prev_y = None, None

    # ── Merge canvas onto frame ───────────────────────────────────────────────
    # Only blend non-black pixels from canvas
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    merged = cv2.add(bg, fg)

    # Draw toolbar on top
    merged = draw_toolbar(merged, current_color_idx, eraser_mode)

    # Status text
    status = f"{'ERASER' if eraser_mode else COLOR_NAMES[current_color_idx]}  |  ☝ draw  ✌ hover"
    cv2.putText(merged, status, (W - 380, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Air Canvas ✋", merged)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()