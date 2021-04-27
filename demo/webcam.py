import cv2
import mediapipe as mp
import numpy as np
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
import collections
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# record time when prev frame is processed
prev_frame_time = 0
# record time when curr frame is processed
curr_frame_time = 0
# Cursor control variables
cursor_control = 'joystick'
clicking_enabled = True
# pyautogui config
pyautogui.FAILSAFE = False


def joystick(center, frame):
    mouse_vector = center - joystick_center
    length = np.linalg.norm(mouse_vector)
    if length > joystick_radius:
        mouse_vector = mouse_vector / length * (length - joystick_radius)
        mouse_move = np.multiply(
            np.power(abs(mouse_vector), 1.75) * 0.05, np.sign(mouse_vector))
        pyautogui.move(tuple(np.int32(mouse_move)), _pause=False)
    cv2.line(frame, tuple(joystick_center), tuple(
        np.int32(center)), (255, 0, 0), 2)


def shoelace_area(points):
    x0, y0 = np.hsplit(points, 2)
    points1 = np.roll(points, -1, axis=0)
    x1, y1 = np.hsplit(points1, 2)
    combination = x0 * y1 - x1 * y0
    area = np.sum(combination) / 2
    return area, x0 + x1, y0 + y1, combination


def palm_center(keypoints, frame):
    indices = [0, 1, 5, 9, 13, 17]
    points = keypoints.reshape((21, 2))
    area, x, y, combination = shoelace_area(points)
    center_x = np.sum(x * combination) / (6 * area)
    center_y = np.sum(y * combination) / (6 * area)
    center = np.array([center_x, center_y])
    radius = int(np.min(np.linalg.norm(points - center, axis=1))
                 * np.mean(frame.shape[:2]))
    center = tuple(np.int32(center * frame.shape[1::-1]))
    return center, radius


def mouse_command(gesture):
    if gesture == 'Eight':
        pyautogui.click()


def map_label_to_gesture(label):
    lab2ges = {
        0: 'Zero',
        1: 'One',
        2: 'Two',
        3: 'Three',
        4: 'Four',
        5: 'Five',
        6: 'Six',
        7: 'Seven',
        8: 'Eight',
        9: 'Nine'
    }

    if label not in lab2ges:
        return 'Unkown'

    return lab2ges[label]


if __name__ == '__main__':
    filename = 'rand_forest.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    hands = mp_hands.Hands(min_detection_confidence=0.5,
                           min_tracking_confidence=0.5, max_num_hands=1)
    cap = cv2.VideoCapture(0)

    # cursor movement initialization
    _, frame = cap.read()
    height, width = frame.shape[:2]
    joystick_center = np.array([int(0.75 * width), int(0.5 * height)])
    joystick_radius = 40

    center_queue = collections.deque(5 * [(0, 0)], 5)

    closed = collections.deque(5 * [0], 5)

    frame_cnt = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                keypoints = [[
                    data_point.x,
                    data_point.y
                ] for data_point in hand_landmarks.landmark]
        keypoints = np.array(keypoints).flatten()

        if frame_cnt % 2 == 0:
            if len(keypoints) != 0:
                # Cursor movement related work
                center, radius = palm_center(keypoints, frame)
                center_queue.appendleft(center)
                center = np.mean(center_queue, axis=0)
                if cursor_control == "joystick":
                    joystick(center, frame)
                cv2.circle(frame, tuple(np.int32(center)), 2, (0, 255, 0), 2)
                cv2.circle(frame, tuple(np.int32(center)),
                           radius, (0, 255, 0), 2)
                cv2.rectangle(frame, (0, 0), (width - 1,
                                              height - 1), (0, 255, 0), 3)

                # Gesture recognition related work
                keypoints[::2] = keypoints[::2] / keypoints[0]
                keypoints[1::2] = keypoints[1::2] / keypoints[1]
                # cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
                probs = loaded_model.predict_proba([keypoints])
                top_prob = np.max(probs)
                if top_prob > 0.2:
                    gesture = map_label_to_gesture(np.argmax(probs))
                else:
                    gesture = 'No Gesture'

                if clicking_enabled:
                    mouse_command(gesture)
            else:
                gesture = 'No Hands Detected'

        # time when we finish processing for this frame
        new_frame_time = time.time()
        # calculate frames per second
        if new_frame_time != prev_frame_time:
            fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        # putting the fps on the frame
        cv2.putText(image, str(fps), (500, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (100, 255, 0), 3, cv2.LINE_AA)
        # putting the recognized gesture on the frame
        cv2.putText(image, str(gesture), (7, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (100, 255, 0), 3, cv2.LINE_AA)

        frame_cnt += 1

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    hands.close()
    cap.release()
