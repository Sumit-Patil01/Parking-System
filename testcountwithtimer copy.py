import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

import anvil.server

anvil.server.connect("server_XEIJCFIIVNN5FJWSSXQJRT3Q-5RJ5O6MGJTR4F6N2")
model = YOLO('yolov8s.pt')

def point_in_polygon_test(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

def draw_polylines(frame, area, index, is_occupied, duration):
    color = (0, 0, 255) if is_occupied else (0, 255, 0)
    cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
    cv2.putText(frame, str(index), (area[0][0], area[0][1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Duration: {duration}", (area[0][0], area[0][1] + 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

def process_frame(frame, areas, timers):
    results = model.predict(frame)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    occupied = [0] * len(areas)
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row)
        class_name = class_list[class_id]
        if 'car' in class_name:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for i, area in enumerate(areas):
                if point_in_polygon_test((cx, cy), area):
                    occupied[i] = 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    cv2.putText(frame, class_name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    if timers[i] is None:
                        timers[i] = time.time()
                    else:
                        duration = time.time() - timers[i]
                        draw_polylines(frame, area, i + 1, True, f"{int(duration)} sec")

    for i, area in enumerate(areas):
        if occupied[i] == 0:
            if timers[i] is not None:
                duration = time.time() - timers[i]
                draw_polylines(frame, area, i + 1, False, f"{int(duration)} sec")
                timers[i] = None

    return frame, occupied, timers

cap = cv2.VideoCapture('parking1.mp4')

with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

areas = [
    [(52, 364), (30, 417), (73, 412), (88, 369)],
    [(105, 353), (86, 428), (137, 427), (146, 358)],
    [(159, 354), (150, 427), (204, 425), (203, 353)],
    [(217, 352), (219, 422), (273, 418), (261, 347)],
    [(274, 345), (286, 417), (338, 415), (321, 345)],
    [(336, 343), (357, 410), (409, 408), (382, 340)],
    [(396, 338), (426, 404), (479, 399), (439, 334)],
    [(458, 333), (494, 397), (543, 390), (495, 330)],
    [(511, 327), (557, 388), (603, 383), (549, 324)],
    [(564, 323), (615, 381), (654, 372), (596, 315)],
    [(616, 316), (666, 369), (703, 363), (642, 312)],
    [(674, 311), (730, 360), (764, 355), (707, 308)],
]

timers = [None] * len(areas)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    frame, occupied, timers = process_frame(frame, areas, timers)

    cv2.imshow("Parking Spaces", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Save data to database or log file here
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    empty_spaces = sum(1 for status in occupied if status == 0)
    available_spaces = sum(1 for status in occupied if status == 1)
    parked_spaces = sum(1 for status in occupied if status == 2)
    print(f"Time: {current_time}, Empty Spaces: {empty_spaces}, Available Spaces: {available_spaces}, Total Spaces: 12")
    @anvil.server.callable   
    def get_data1():
        data1 = "12"
        return  data1
    @anvil.server.callable
    def get_data2():
        data2 = available_spaces
        return  data2
    @anvil.server.callable
    def get_data3():
        data3 = empty_spaces
        return  data3     

    time.sleep(1)  # Wait for 2 seconds before processing the next frame

cap.release()
cv2.destroyAllWindows()
anvil.server.wait_forever()
