import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from regular import predict as regular_predict

# ---------- MediaPipe 初始化 ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ---------- 多页字母映射（只使用手势0-2） ----------
PAGES = [
    ['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i'], ['j', 'k', 'l'],
    ['m', 'n', 'o'], ['p', 'q', 'r'], ['s', 't', 'u'], ['v', 'w', 'x'], ['y', 'z', '']
]

current_page = 0
last_class = -1
stable_count = 0
STABLE_THRESHOLD = 5
last_keypress_time = 0
KEYPRESS_DELAY = 1.5
last_page_switch_time = 0
PAGE_SWITCH_DELAY = 1.0
last_x_position = None
last_y_position = None
leave_time = 0
LEAVE_TRIGGER_DELAY = 0.5
frame_center_ratio = 0.4

typed_text = ""
pending_char = ""
pending_start_time = 0
confirm_cooldown = 0
delete_cooldown = 0
CONFIRM_DELAY = 1.0
DELETE_DELAY = 1.0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    current_time = time.time()
    h, w, _ = frame.shape

    current_label = f"Page: {current_page + 1}\nText: {typed_text}"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                x = lm.x * w
                y = lm.y * h
                keypoints.append([x, y, 0.0])

            input_array = np.array(keypoints, dtype=np.float32)
            pred_class = regular_predict(input_array)

            hand_x = keypoints[0][0]
            hand_y = keypoints[0][1]
            last_x_position = hand_x
            last_y_position = hand_y
            leave_time = 0

            if hand_y < h * 0.1 and current_time - delete_cooldown > DELETE_DELAY:
                if typed_text:
                    typed_text = typed_text[:-1]
                    print("[DELETE] ←")
                    delete_cooldown = current_time
                continue

            if pred_class == 11:
                if pending_char and current_time - confirm_cooldown > CONFIRM_DELAY:
                    typed_text += pending_char
                    print(f"[CONFIRM] {pending_char}")
                    pending_char = ""
                    confirm_cooldown = current_time
                continue

            if pred_class == 3:
                predicted_key = ' '
                display_key = 'space'
            elif pred_class > 2:
                continue
            else:
                predicted_key = PAGES[current_page][pred_class] if pred_class < len(PAGES[current_page]) else ''
                display_key = predicted_key

            if pred_class == last_class:
                stable_count += 1
            else:
                stable_count = 0
                last_class = pred_class

            if stable_count == STABLE_THRESHOLD and current_time - last_keypress_time > KEYPRESS_DELAY:
                if predicted_key is not None:
                    pending_char = predicted_key
                    pending_start_time = current_time
                    print(f"[PENDING] {display_key}")
                    last_keypress_time = current_time
                stable_count = 0

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 更新 current_label 内部
            display_pending = 'space' if pending_char == ' ' else pending_char
            display_prediction = 'space' if predicted_key == ' ' else predicted_key
            current_label += f"\nPending: {display_pending}\nPrediction: {display_prediction}"

    else:
        if last_x_position is not None and current_time - leave_time > LEAVE_TRIGGER_DELAY:
            if last_y_position is not None and last_y_position > h * 0.9 and current_time - delete_cooldown > DELETE_DELAY:
                if typed_text:
                    typed_text = typed_text[:-1]
                    print("[DELETE - DOWN LEAVE] ←")
                    delete_cooldown = current_time
            elif last_x_position < w * (1 - frame_center_ratio):
                current_page = (current_page - 1 + len(PAGES)) % len(PAGES)
                print(f"[PAGE SWITCH ← LEFT LEAVE] → Page {current_page + 1}")
            elif last_x_position > w * frame_center_ratio:
                current_page = (current_page + 1) % len(PAGES)
                print(f"[PAGE SWITCH → RIGHT LEAVE] → Page {current_page + 1}")
            leave_time = current_time
            last_x_position = None

    # 清除超时未确认的 pending
    if pending_char and current_time - pending_start_time > 1.0:
        print(f"[PENDING CLEARED] {pending_char}")
        pending_char = ""

    # 显示 current_label 多行文本
    y0 = 40
    for i, line in enumerate(current_label.split("\n")):
        y = y0 + i * 30
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    frame_resized = cv2.resize(frame, None, fx=1.5, fy=1.5)
    cv2.imshow("Hand Gesture Control", frame_resized)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
