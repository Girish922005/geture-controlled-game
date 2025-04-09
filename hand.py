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

# Game control variables
throttle_pressed = False
brake_pressed = False
last_gesture = None
gesture_cooldown = 0.5  # Cooldown time in seconds
last_gesture_time = 0

def calculate_finger_angles(hand_landmarks):
    """Calculate angles between finger segments"""
    # Get finger landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Get finger base positions
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    # Calculate angles for each finger
    angles = {
        'index': np.arctan2(index_tip.y - index_base.y, index_tip.x - index_base.x),
        'middle': np.arctan2(middle_tip.y - middle_base.y, middle_tip.x - middle_base.x),
        'ring': np.arctan2(ring_tip.y - ring_base.y, ring_tip.x - ring_base.x),
        'pinky': np.arctan2(pinky_tip.y - pinky_base.y, pinky_tip.x - pinky_base.x)
    }
    
    return angles

def detect_gesture(hand_landmarks):
    """Detect hand gestures based on hand landmarks"""
    # Get finger landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Get finger base positions
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    # Calculate finger angles
    angles = calculate_finger_angles(hand_landmarks)
    
    # Thumbs up detection (thumb pointing up)
    if thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y:
        return "throttle"
    
    # Thumbs down detection (thumb pointing down)
    if thumb_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y:
        return "brake"
    
    # Victory sign detection (index and middle fingers up, others down)
    if (abs(angles['index']) < 0.5 and  # Close to vertical
        abs(angles['middle']) < 0.5 and  # Close to vertical
        abs(angles['ring']) > 1.0 and    # More horizontal
        abs(angles['pinky']) > 1.0):     # More horizontal
        return "lean_forward"
    
    # Fist detection (all fingers down)
    if (abs(angles['index']) > 1.0 and   # More horizontal
        abs(angles['middle']) > 1.0 and  # More horizontal
        abs(angles['ring']) > 1.0 and    # More horizontal
        abs(angles['pinky']) > 1.0):     # More horizontal
        return "lean_backward"
    
    return None

def handle_game_controls(gesture):
    """Handle game controls based on detected gesture"""
    global throttle_pressed, brake_pressed, last_gesture, last_gesture_time
    
    current_time = time.time()
    
    # Check if enough time has passed since last gesture
    if current_time - last_gesture_time < gesture_cooldown:
        return
    
    # Handle throttle (thumbs up)
    if gesture == "throttle" and not throttle_pressed:
        keyboard.press(Key.right)
        throttle_pressed = True
        last_gesture = "throttle"
        last_gesture_time = current_time
    elif gesture != "throttle" and throttle_pressed:
        keyboard.release(Key.right)
        throttle_pressed = False
    
    # Handle brake (thumbs down)
    if gesture == "brake" and not brake_pressed:
        keyboard.press(Key.left)
        brake_pressed = True
        last_gesture = "brake"
        last_gesture_time = current_time
    elif gesture != "brake" and brake_pressed:
        keyboard.release(Key.left)
        brake_pressed = False
    
    # Handle lean forward (victory sign)
    if gesture == "lean_forward" and last_gesture != "lean_forward":
        keyboard.press(Key.up)
        time.sleep(0.1)  # Press briefly
        keyboard.release(Key.up)
        last_gesture = "lean_forward"
        last_gesture_time = current_time
    
    # Handle lean backward (fist)
    if gesture == "lean_backward" and last_gesture != "lean_backward":
        keyboard.press(Key.down)
        time.sleep(0.1)  # Press briefly
        keyboard.release(Key.down)
        last_gesture = "lean_backward"
        last_gesture_time = current_time

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    print("Hill Climb Racing Gesture Controls:")
    print("Thumbs Up: Throttle (Right Arrow)")
    print("Thumbs Down: Brake (Left Arrow)")
    print("Victory Sign (✌️): Lean Forward (Up Arrow)")
    print("Fist: Lean Backward (Down Arrow)")
    print("Press 'q' to quit")
    
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
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Detect gesture and handle game controls
                gesture = detect_gesture(hand_landmarks)
                if gesture:
                    handle_game_controls(gesture)
                    # Display current gesture
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Hill Climb Racing - Gesture Controls', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
    