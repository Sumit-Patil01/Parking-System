import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import random

import anvil.server

anvil.server.connect("server_JSCAUAQJ7VT3JBZZJWJS5EJH-B6QKOMYKSQK7SQ4C")
model = YOLO('yolov8s.pt')

def point_in_polygon_test(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

def draw_polylines(frame, area, index, is_occupied, is_incorrectly_parked=False):
    color = (0, 0, 255) if is_occupied else (0, 255, 0)
    if is_incorrectly_parked:
        color = (0, 165, 255)  # Orange color for incorrectly parked
    cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
    cv2.putText(frame, str(index), (area[0][0], area[0][1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

def is_parked_incorrectly(bbox, area):
    # Simple heuristic: If the bounding box center is too far from the area's center or dimensions are wrong
    (x1, y1, x2, y2) = bbox
    bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    area_center = np.mean(area, axis=0).astype(int)
    
    # Calculate distance between the centers
    dist = np.linalg.norm(bbox_center - area_center)
    # Define a threshold for incorrect parking
    max_distance = 30  # Adjust this threshold based on actual data
    
    # Additional checks for dimensions can be added here
    
    return dist > max_distance

def process_frame(frame, areas):
    results = model.predict(frame)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    occupied = [0] * len(areas)
    incorrectly_parked = [0] * len(areas)
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row)
        class_name = class_list[class_id]
        if 'car' in class_name:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for i, area in enumerate(areas):
                if point_in_polygon_test((cx, cy), area):
                    occupied[i] = 1
                    if is_parked_incorrectly((x1, y1, x2, y2), area):
                        incorrectly_parked[i] = 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    cv2.putText(frame, class_name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    empty_spaces = []
    for i, status in enumerate(occupied):
        if status == 0:
            empty_spaces.append(i + 1)

    for i, area in enumerate(areas):
        draw_polylines(frame, area, i + 1, occupied[i], incorrectly_parked[i])
        
    for space in empty_spaces:
        text = f"Space {space} available"
        cv2.putText(frame, text, (10, 30 * space), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Empty Spaces: {len(empty_spaces)}", (10, 30 * (len(areas) + 1)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Parked Spaces: {len(areas) - len(empty_spaces)}", (10, 30 * (len(areas) + 2)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Incorrectly Parked: {sum(incorrectly_parked)}", (10, 30 * (len(areas) + 3)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    return frame, len(empty_spaces), len(areas) - len(empty_spaces), empty_spaces, incorrectly_parked

cap = cv2.VideoCapture('CAMP2.mp4')

with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

areas = [
    [(634,310), (582,391), (829,459), (783,353)],
    [(505,305), (438,359), (544,390), (609,332)],
    [(503,287), (417,339), (328,324), (420,280) ],
    [(802,370), (824,467), (987,468), (968,438)],
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    frame, empty_spaces, parked_spaces, empty_spaces_list, incorrectly_parked_list = process_frame(frame, areas)

    cv2.imshow("Parking Spaces", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    if empty_spaces_list:
        random_free_spot = random.choice(empty_spaces_list)
        print(f"Time: {current_time}, Empty Spaces: {empty_spaces}, Parked Spaces: {parked_spaces}, Total Spaces: 12, Random Free Spot: {random_free_spot}")
    else:
        random_free_spot = None
        print(f"Time: {current_time}, Empty Spaces: {empty_spaces}, Parked Spaces: {parked_spaces}, Total Spaces: 12, No Free Spots Available")

    incorrectly_parked_spots = [i+1 for i, val in enumerate(incorrectly_parked_list) if val]
    if incorrectly_parked_spots:
        print(f"Incorrectly Parked Spots: {incorrectly_parked_spots}")

    @anvil.server.callable   
    def get_data():
        data = {
            "total_spaces": 4,
            "empty_spaces": empty_spaces,
            "parked_spaces": parked_spaces,
            "random_free_spot": random_free_spot,
            "incorrectly_parked_spots": incorrectly_parked_spots
        }
        return data  


cap.release()
cv2.destroyAllWindows()
anvil.server.wait_forever()
