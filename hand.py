import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Key, Controller
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize keyboard controller
keyboard = Controller()

# Initialize game variables
game_object_x = 400  # Starting X position
game_object_y = 300  # Starting Y position
object_size = 50
movement_speed = 10

def detect_gesture(hand_landmarks):
    """Detect hand gestures based on hand landmarks"""
    # Get thumb tip and index finger tip
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    # Thumbs up detection (thumb pointing up)
    if thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y:
        return "up"
    
    # Thumbs down detection (thumb pointing down)
    if thumb_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y:
        return "down"
    
    # Victory sign detection (index and middle fingers up)
    if (index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
        middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y):
        return "right"
    
    # Fist detection (all fingers down)
    fingers_down = all(
        hand_landmarks.landmark[i].y > hand_landmarks.landmark[i-2].y
        for i in [8, 12, 16, 20]  # Index, middle, ring, and pinky tips
    )
    if fingers_down:
        return "left"
    
    return None

def update_game_object(gesture):
    """Update game object position based on gesture"""
    global game_object_x, game_object_y
    
    if gesture == "up":
        game_object_y -= movement_speed
    elif gesture == "down":
        game_object_y += movement_speed
    elif gesture == "right":
        game_object_x += movement_speed
    elif gesture == "left":
        game_object_x -= movement_speed
    
    # Keep object within screen bounds
    game_object_x = max(0, min(game_object_x, 800 - object_size))
    game_object_y = max(0, min(game_object_y, 600 - object_size))

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_frame)
        
        # Draw game object
        cv2.rectangle(frame, 
                     (game_object_x, game_object_y),
                     (game_object_x + object_size, game_object_y + object_size),
                     (0, 255, 0), -1)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Detect gesture and update game object
                gesture = detect_gesture(hand_landmarks)
                if gesture:
                    update_game_object(gesture)
        
        # Display the frame
        cv2.imshow('Gesture-Controlled Game', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
    