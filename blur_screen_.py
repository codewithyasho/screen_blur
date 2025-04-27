import cv2
import pyautogui
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageFilter, ImageTk
import tkinter as tk
from tkinter import Label
import threading

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model (small and fast)

# Create a GUI window for the blur effect
root = tk.Tk()
root.attributes('-fullscreen', True)
root.configure(bg="black")
label = Label(root, bg="black")
label.pack(fill="both", expand=True)
root.withdraw()  # Hide initially

# Capture video from webcam
cap = cv2.VideoCapture(0)

def detect_people(frame):
    """Detects people in the frame using YOLOv8."""
    results = model(frame)  # Run YOLO detection
    people_count = sum(1 for obj in results[0].boxes.cls if obj == 0)  # Count 'person' detections
    return people_count

def blur_screen():
    """Captures and blurs the screen, then displays it over everything."""
    screenshot = pyautogui.screenshot()
    img = screenshot.filter(ImageFilter.GaussianBlur(20))  # Apply blur
    img = img.resize((root.winfo_screenwidth(), root.winfo_screenheight()))
    img_tk = ImageTk.PhotoImage(img)

    label.config(image=img_tk)
    label.image = img_tk
    root.deiconify()  # Show blur screen

def normal_screen():
    """Hides the blur overlay."""
    root.withdraw()

def process_video():
    """Main loop to process the webcam feed."""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        people_count = detect_people(frame)
        print("People Detected:", people_count)

        if people_count >= 2:
            blur_screen()
        else:
            normal_screen()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Run video processing in a separate thread
threading.Thread(target=process_video, daemon=True).start()
root.mainloop()  # Start GUI loop
