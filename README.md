# Gesture-Controlled Game

This project implements a simple gesture-controlled game using Python, OpenCV, MediaPipe, and pynput. The game allows you to control a simple object using hand gestures.

## Setup Instructions

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the game:
```bash
python hand.py
```

## Gesture Controls

- Thumbs Up: Move object up
- Thumbs Down: Move object down
- Victory Sign (✌️): Move object right
- Fist: Move object left
- Open Palm: Reset object position

## Requirements

- Python 3.8 or higher
- Webcam
- Required Python packages (listed in requirements.txt) 