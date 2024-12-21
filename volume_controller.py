import cv2  
import mediapipe as mp  
import pyautogui 
camera = cv2.VideoCapture(0)  # Start video capture using the default webcam
drawing_utils = mp.solutions.drawing_utils  # Get tools for drawing hand landmarks
hands = mp.solutions.hands.Hands()  # Create hand tracking model to detect hands
while True:
    _, image = camera.read()  # Read a frame from the camera
    image = cv2.flip(image, 1)  # Flip the image to create a mirror effect
    height, width, _ = image.shape  # Get the size of the image (height and width)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image from BGR to RGB for MediaPipe
    output = hands.process(rgb_image)  # Process the image to detect hands
    my_hands = output.multi_hand_landmarks  # Get hand landmarks if any hands are detected
    if my_hands:  # If hands are detected in the image
        for hand in my_hands:  # Loop through detected hands
            drawing_utils.draw_landmarks(image, hand)  # Draw hand landmarks on the image
            landmarks = hand.landmark  # Get the list of landmarks for the hand
            x1, y1, x2, y2 = None, None, None, None  # Initialize positions of finger landmarks
            for id, landmark in enumerate(landmarks):  # Loop through each landmark of the hand
                x = int(landmark.x * width)  # Convert normalized x-coordinate to actual pixel value
                y = int(landmark.y * height)  # Convert normalized y-coordinate to actual pixel value
                if id == 8:  # If the landmark is the tip of the index finger (id 8)
                    cv2.circle(img=image, center=(x, y), radius=10, color=(255, 255, 255), thickness=5)  # Draw a white circle
                    x1, y1 = x, y  # Store the coordinates of the index finger tip
                if id == 4:  # If the landmark is the base of the thumb (id 4)
                    cv2.circle(img=image, center=(x, y), radius=10, color=(0, 0, 0), thickness=5)  # Draw a black circle
                    x2, y2 = x, y  # Store the coordinates of the thumb base
                if x1 is not None and y1 is not None and x2 is not None and y2 is not None:  # If both finger positions are found
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 5)  # Draw a line between the thumb base and index finger tip
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 4  # Calculate distance between thumb base and index tip
        if length > 10:  # If the distance is large, simulate pressing the "Volumeup" key
            pyautogui.press("volumeup")
        else:  # If the distance is small, simulate pressing the "Volumedown" key
            pyautogui.press("volumedown")
    cv2.imshow("Volume Control Using Computer Vision", image)  # Show the processed image with hand landmarks
    key = cv2.waitKey(10)  # Wait for a key press for 10 milliseconds
    if key == 27:  # If the escape key (ASCII 27) is pressed, exit the loop
        break  # Exit the loop and end the program
camera.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close all OpenCV windows
