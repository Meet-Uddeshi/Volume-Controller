# Step 1 => Importing necessary libraries
import cv2
import mediapipe as mp
import pyautogui

# Step 2 => Starting video capture and initializing hand tracking
camera = cv2.VideoCapture(0)
drawing_utils = mp.solutions.drawing_utils
hands = mp.solutions.hands.Hands()

# Step 3 => Main loop for processing frames
while True:
    # Step 4 => Reading a frame, flipping it, and getting image dimensions
    _, image = camera.read()
    image = cv2.flip(image, 1)
    height, width, _ = image.shape

    # Step 5 => Converting image to RGB and processing for hand detection
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = hands.process(rgb_image)
    my_hands = output.multi_hand_landmarks

    # Step 6 => Hand detection and landmark processing
    if my_hands:
        for hand in my_hands:
            drawing_utils.draw_landmarks(image, hand)
            landmarks = hand.landmark
            x1, y1, x2, y2 = None, None, None, None  # Initialize finger landmark positions

            # Step 7 => Iterating through landmarks and identifying thumb and index finger tips
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                if id == 8:  # Index finger tip
                    cv2.circle(img=image, center=(x, y), radius=10, color=(255, 255, 255), thickness=5)
                    x1, y1 = x, y
                if id == 4:  # Thumb base
                    cv2.circle(img=image, center=(x, y), radius=10, color=(0, 0, 0), thickness=5)
                    x2, y2 = x, y

                # Step 8 => Drawing line between thumb and index finger if both are detected
                if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 5)

            # Step 9 => Calculating distance and controlling volume
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 4
            if length > 10:
                pyautogui.press("volumeup")
            else:
                pyautogui.press("volumedown")

    # Step 10 => Displaying the image and handling key press
    cv2.imshow("Volume Control Using Computer Vision", image)
    key = cv2.waitKey(10)
    if key == 27:  # Escape key
        break

# Step 11 => Releasing resources and closing windows
camera.release()
cv2.destroyAllWindows()
