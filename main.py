# TODO: Make a probability distribution bar

import cv2
import numpy as np
from model import My2LP

model = My2LP("./data/mnist_weights.npz")
canvas = np.zeros((420, 420), dtype=np.uint8)

# State variables
drawing = False
mouse_x, mouse_y = 0, 0
margin = 60

def update_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

cv2.namedWindow("MNIST Predictor")
cv2.setMouseCallback("MNIST Predictor", update_mouse)

print("Controls: [Space] Toggle Pen | [C] Clear | [Q] Quit")

while True:
    if drawing:
        cv2.circle(canvas, (mouse_x, mouse_y), 12, 255, -1)

    # PREPROCESSING & INFERENCE
    prediction = "None"
    coords = cv2.findNonZero(canvas)
    
    if coords is not None:
        # crop the digit
        x, y, w, h = cv2.boundingRect(coords)
        digit_roi = canvas[y:y+h, x:x+w]
        
        # Padding
        digit_roi = cv2.copyMakeBorder(digit_roi, margin, margin, margin, margin, 
                                       cv2.BORDER_CONSTANT, value=0)
        
        # Resize (28x28)
        resized = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize & flatten to (784, 1)
        input_data = (resized.astype(np.float32) / 255.0).reshape(784, 1)
        
        prediction = model.inference(input_data)

    # Render UI
    display_img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    
    # Visual indicator if the pen is active
    status_color = (0, 255, 0) if drawing else (0, 0, 255)
    status_text = "PEN: ON" if drawing else "PEN: OFF"
    
    cv2.putText(display_img, f"{status_text} | Pred: {prediction}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    cv2.imshow("MNIST Predictor", display_img)

    # 4. KEYBOARD CONTROLS
    key = cv2.waitKey(1) & 0xFF
    if key == 32: # Spacebar
        drawing = not drawing
    elif key == ord('c'):
        canvas[:] = 0
    elif key == ord('q'):
        break

cv2.destroyAllWindows()