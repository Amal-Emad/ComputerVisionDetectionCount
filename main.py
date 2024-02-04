import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)

# Initialize YOLO model
model = YOLO('yolov8s.pt')

# Mouse function to detect region of Interest
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open video capture
cap = cv2.VideoCapture('peoplecount1.mp4')

# Read class names from coco.txt
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Initialize count, tracker, and sets
count = 0
tracker = Tracker()
people_entering = {}
people_exiting = {}
entering = set()
exiting = set()

# Define the two areas to detect entering and exiting
area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]

while True:
    # Read the frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    # Get predictions from the YOLO model
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # List to store person bounding boxes
    bbox_list = []

    # Iterate through YOLO predictions
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            bbox_list.append([x1, y1, x2, y2])

    # Update the tracker with the current frame's bounding boxes
    bbox_id = tracker.update(bbox_list)

    # Process each bounding box
    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id = bbox
        results_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
        
        # Check if the person is in area2 (Entering)
        if results_area2 >= 0:
            people_entering[obj_id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), COLOR_YELLOW, 2)

        if obj_id in people_entering:
            results_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
            
            # Check if the person is in area1 (Inside)
            if results_area1 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), COLOR_GREEN, 2)
                cv2.circle(frame, (x4, y4), 5, COLOR_MAGENTA, -1)
                cv2.putText(frame, str(obj_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLOR_YELLOW, 1)
                entering.add(obj_id)

        results_area1_exit = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
        
        # Check if the person is in area1 (Exiting)
        if results_area1_exit >= 0:
            people_exiting[obj_id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), COLOR_GREEN, 2)

        if obj_id in people_exiting:
            results_area2_exit = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
            
            # Check if the person is in area2 (Outside)
            if results_area2_exit >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), COLOR_MAGENTA, 2)
                cv2.circle(frame, (x4, y4), 5, COLOR_MAGENTA, -1)
                cv2.putText(frame, str(obj_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLOR_YELLOW, 1)
                exiting.add(obj_id)

    # Draw the polylines for areas
    cv2.polylines(frame, [np.array(area1, np.int32)], True, COLOR_BLUE, 2)
    cv2.putText(frame, str('1'), (504, 471), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLOR_BLUE, 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, COLOR_BLUE, 2)
    cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLOR_BLUE, 1)

    # Display counts and additional information
    enter = len(entering)
    exit_count = len(exiting)
    cv2.putText(frame, f"Enter: {enter}", (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, COLOR_MAGENTA, 1)
    cv2.putText(frame, f"Exit: {exit_count}", (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.8, COLOR_MAGENTA, 1)
    # Draw a white rectangle as background
    cv2.rectangle(frame, (890, 480 - 20), (890 + 150, 480 + 10), (255, 255, 255), -1)

    # Add text with a blue color
    cv2.putText(frame, str('AMAL-Camera'), (890, 480), cv2.FONT_HERSHEY_COMPLEX, 0.6, COLOR_BLUE, 1)


    # Show the frame
    cv2.imshow("RGB", frame)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
