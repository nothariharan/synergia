import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import torch
import time  
import csv 
from datetime import datetime 

print(f"--- Is CUDA (GPU) available? {torch.cuda.is_available()} ---")

# --- 1. configuration ---

# === FOR VIDEO FILE (Current) ===
# VIDEO_FILE = 'test2_clean.mp4'  # input video

# === FOR LIVE CAMERA (Future) ===
# (To switch to live, comment out the line above and uncomment the line below)
VIDEO_FILE = 0  # 0 is the default webcam. Try 1 or 2 if you have multiple cameras.


# measured heights of objects in inches
KNOWN_HEIGHTS_INCHES = {
    'Cable': 60.0,
    'Ball_Red': 2.575,
    'Ball_Blue': 2.575,
    'Rock': 4,
    'Bottle': 8,
    'Box': 5,
    'DockingStation': 31,
    'Thread': 50,
    'Other': 10
}

# --- phase 3 risk score configuration ---
RISK_LEVELS = {
    'Cable': 1.0,           
    'Thread': 1.0,          
    'Rock': 0.7,            
    'DockingStation': 0.7,  
    'Box': 0.6,             
    'Bottle': 0.3,          
    'Ball_Red': 0.1,        
    'Ball_Blue': 0.1,       
    'Other': 0.5            
}
RISK_COLOR_LOW = (0, 255, 0)     
RISK_COLOR_MEDIUM = (0, 255, 255) 
RISK_COLOR_HIGH = (0, 0, 255)     

# --- 720p processing ---
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
HORIZONTAL_FOV = 80.0
FOCAL_LENGTH_PIXELS = 762 

# model setup
MODEL_FILE = 'yolov8n.pt'
CONF_THRESHOLD = 0.13

# --- 2. load assets ---

print(f"Loading model: {MODEL_FILE}")
model = YOLO(MODEL_FILE)
CLASS_NAMES = model.names

distance_history = defaultdict(lambda: [])
SMOOTHING_WINDOW = 5
TRACK_MEMORY_FRAMES = 5  
tracked_objects_cache = {} 

# --- 3. helper functions (distance/bearing) ---
def estimate_bearing(object_center_x):
    X = object_center_x - (IMAGE_WIDTH / 2)
    angle_rad = np.arctan(X / FOCAL_LENGTH_PIXELS)
    return np.degrees(angle_rad)

def estimate_distance(object_pixel_height, class_name):
    class_name_upper = class_name.upper()
    known_height = None
    for key in KNOWN_HEIGHTS_INCHES:
        if key.upper() == class_name_upper:
            known_height = KNOWN_HEIGHTS_INCHES[key]
            break
    if known_height:
        return (known_height * FOCAL_LENGTH_PIXELS) / object_pixel_height
    return None

def get_smoothed_distance(track_id, distance_inches):
    history = distance_history[track_id]
    history.append(distance_inches)
    if len(history) > SMOOTHING_WINDOW:
        history.pop(0)
    return np.mean(history)

# --- phase 3 risk function ---
def calculate_risk(class_name, distance_inches):
    base_risk = RISK_LEVELS.get(class_name, 0.5) 
    
    if distance_inches is None or distance_inches <= 0:
        return 0.0, "Risk: N/A", RISK_COLOR_LOW 

    risk_score = (base_risk / (distance_inches / 60))
    
    if risk_score > 1.0:
        return risk_score, "Risk: HIGH", RISK_COLOR_HIGH
    elif risk_score > 0.6:
        return risk_score, "Risk: MEDIUM", RISK_COLOR_MEDIUM
    else:
        return risk_score, "Risk: LOW", RISK_COLOR_LOW

# --- 4. playback and button controls ---

playback_mode = 'normal'  
frame_counter = 0
FRAME_SKIP = 5
FRAME_SKIP_SUPER = 20 
WINDOW_NAME = 'YOLOv8 Underwater System' 

BTN_Y = 660  
BTN_H = 50   
BTN_W = 150  
BTN_MARGIN = 10
BTN_PAUSE = [BTN_MARGIN, BTN_Y, BTN_MARGIN + BTN_W, BTN_Y + BTN_H]
BTN_NORMAL = [BTN_MARGIN*2 + BTN_W, BTN_Y, BTN_MARGIN*2 + BTN_W*2, BTN_Y + BTN_H]
BTN_FAST = [BTN_MARGIN*3 + BTN_W*2, BTN_Y, BTN_MARGIN*3 + BTN_W*3, BTN_Y + BTN_H]
BTN_SUPER_FAST = [BTN_MARGIN*4 + BTN_W*3, BTN_Y, BTN_MARGIN*4 + BTN_W*4, BTN_Y + BTN_H]

def draw_buttons(frame):
    global playback_mode
    btn_color = (80, 80, 80) 
    text_color = (255, 255, 255)
    active_color = (0, 200, 0)
    
    pause_c = active_color if playback_mode == 'pause' else btn_color
    normal_c = active_color if playback_mode == 'normal' else btn_color
    fast_c = active_color if playback_mode == 'fast' else btn_color
    super_fast_c = active_color if playback_mode == 'super_fast' else btn_color

    cv2.rectangle(frame, (BTN_PAUSE[0], BTN_PAUSE[1]), (BTN_PAUSE[2], BTN_PAUSE[3]), pause_c, -1)
    cv2.putText(frame, "PAUSE (p)", (BTN_PAUSE[0] + 10, BTN_PAUSE[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.rectangle(frame, (BTN_NORMAL[0], BTN_NORMAL[1]), (BTN_NORMAL[2], BTN_NORMAL[3]), normal_c, -1)
    cv2.putText(frame, "NORMAL (n)", (BTN_NORMAL[0] + 10, BTN_NORMAL[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.rectangle(frame, (BTN_FAST[0], BTN_FAST[1]), (BTN_FAST[2], BTN_FAST[3]), fast_c, -1)
    cv2.putText(frame, "FAST (f)", (BTN_FAST[0] + 10, BTN_FAST[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.rectangle(frame, (BTN_SUPER_FAST[0], BTN_SUPER_FAST[1]), (BTN_SUPER_FAST[2], BTN_SUPER_FAST[3]), super_fast_c, -1)
    cv2.putText(frame, "SUPER FAST", (BTN_SUPER_FAST[0] + 10, BTN_SUPER_FAST[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

def mouse_callback(event, x, y, flags, param):
    global playback_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        if BTN_PAUSE[0] <= x <= BTN_PAUSE[2] and BTN_PAUSE[1] <= y <= BTN_PAUSE[3]:
            playback_mode = 'pause' if playback_mode != 'pause' else 'normal'
        elif BTN_NORMAL[0] <= x <= BTN_NORMAL[2] and BTN_NORMAL[1] <= y <= BTN_NORMAL[3]:
            playback_mode = 'normal'
        elif BTN_FAST[0] <= x <= BTN_FAST[2] and BTN_FAST[1] <= y <= BTN_FAST[3]:
            playback_mode = 'fast' if playback_mode != 'fast' else 'normal'
        elif BTN_SUPER_FAST[0] <= x <= BTN_SUPER_FAST[2] and BTN_SUPER_FAST[1] <= y <= BTN_SUPER_FAST[3]:
            playback_mode = 'super_fast' if playback_mode != 'super_fast' else 'normal'

# --- 5. main processing loop ---

# --- new: intelligent camera/file opening ---
if isinstance(VIDEO_FILE, int):
    # it's a live camera
    cap = cv2.VideoCapture(VIDEO_FILE + cv2.CAP_DSHOW)
    # request 720p from the camera for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # set buffer size
else:
    # it's a video file
    cap = cv2.VideoCapture(VIDEO_FILE)
    # do not add CAP_DSHOW or set buffer

if not cap.isOpened():
    if isinstance(VIDEO_FILE, int):
        print(f"Error: Could not open camera with index {VIDEO_FILE}.")
        print("1. Is it plugged in? \n2. Is it used by another app (Zoom, Teams)? \n3. Did your 'find_camera.py' script find a different index (like 1)?")
    else:
        print(f"Error: Could not open video file {VIDEO_FILE}")
    exit()
# --- end of new block ---

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) 
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
cv2.resizeWindow(WINDOW_NAME, IMAGE_WIDTH, IMAGE_HEIGHT) 

# --- setup csv file for logging ---
# (this logic is now separated from the camera opening)
# === FOR VIDEO FILE (Current) ===
csv_file_name = 'detection_log.csv'

# === FOR LIVE CAMERA (Future) ===
# (comment out the line above and uncomment the line below to create unique logs)
# current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# csv_file_name = f'log_{current_time}.csv'

try:
    csv_file = open(csv_file_name, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'frame', 'track_id', 'class_name', 'confidence', 
        'distance_inches', 'bearing_degrees', 'risk_score', 'risk_label',
        'pixel_width', 'pixel_height'
    ])
    print(f"Logging detections to {csv_file_name}")
except IOError as e:
    print(f"Error: Could not open CSV file for writing: {e}")
    csv_writer = None 

print("\n--- CONTROLS ---")
print("  Clickable buttons on-screen")
print("  [q] = Quit (keyboard)")
print(f"\nProcessing {'Live Camera' if isinstance(VIDEO_FILE, int) else VIDEO_FILE}...")

prev_frame_time = 0
fps_to_display = 0

while cap.isOpened():
    if playback_mode == 'pause':
        key = cv2.waitKey(0)  
    else:
        key = cv2.waitKey(1)  

    if key & 0xFF == ord('q'):
        break
    
    if playback_mode == 'pause':
        draw_buttons(frame)
        cv2.imshow(WINDOW_NAME, frame)
        continue

    ret, frame = cap.read()
    if not ret:
        if isinstance(VIDEO_FILE, int):
            print("Error: Camera feed lost. Exiting.")
        else:
            print("End of video file.")
        break 
        
    # resize frame to 720p *before* processing
    frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

    frame_counter += 1

    if (playback_mode == 'fast' and frame_counter % FRAME_SKIP != 0) or \
       (playback_mode == 'super_fast' and frame_counter % FRAME_SKIP_SUPER != 0):
        draw_buttons(frame)
        fps_text = f"FPS: {fps_to_display:.1f} (Skipping)"
        cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(WINDOW_NAME, frame) 
        continue 

    results = model.track(frame,
                          persist=True,
                          verbose=False,
                          conf=CONF_THRESHOLD,
                          imgsz=640) 

    seen_track_ids = set() 
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            track_id = track_ids[i]
            seen_track_ids.add(track_id) 
            
            tracked_objects_cache[track_id] = {
                'box': boxes[i],
                'conf': confidences[i],
                'class_id': class_ids[i],
                'last_seen_frame': frame_counter 
            }

    dead_tracks = [] 
    for track_id, data in tracked_objects_cache.items():
        
        draw_object = False
        color = (0, 255, 0) 

        if track_id in seen_track_ids:
            draw_object = True
        else:
            if playback_mode == 'normal':
                dead_tracks.append(track_id)
                continue 
            else: 
                if frame_counter - data['last_seen_frame'] <= TRACK_MEMORY_FRAMES:
                    draw_object = True 
                    color = (0, 100, 0) 
                else:
                    dead_tracks.append(track_id)
                    continue

        if draw_object: 
            x1, y1, x2, y2 = data['box']
            conf = data['conf']
            class_id = data['class_id']
            class_name = CLASS_NAMES.get(class_id, 'Unknown')
            
            pixel_height = y2 - y1
            pixel_width = x2 - x1
            object_center_x = (x1 + x2) / 2
            
            angle = estimate_bearing(object_center_x)
            distance = estimate_distance(pixel_height, class_name)

            if distance:
                stable_distance = get_smoothed_distance(track_id, distance)
                dist_label = f"Dist: {stable_distance:.2f} inches"
            else:
                stable_distance = None 
                dist_label = "Dist: N/A"

            risk_score, risk_label, risk_color = calculate_risk(class_name, stable_distance)
            
            if track_id in seen_track_ids:
                color = risk_color 
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_name = f"{class_name.upper()}: {conf:.2f}"
            label_dist = dist_label
            label_angle = f"Angle: {angle:.2f} degrees"
            label_size = f"Size: {pixel_width}x{pixel_height} px"
            label_risk = risk_label 
            
            y_offset = y2 + 20 
            
            cv2.putText(frame, label_name, (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, label_dist, (x1, y_offset + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, label_angle, (x1, y_offset + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, label_size, (x1, y_offset + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, label_risk, (x1, y_offset + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if csv_writer is not None:
                csv_writer.writerow([
                    frame_counter, track_id, class_name, f"{conf:.2f}",
                    f"{stable_distance:.2f}" if stable_distance is not None else "N/A",
                    f"{angle:.2f}", f"{risk_score:.2f}", risk_label.replace("Risk: ", ""),
                    pixel_width, pixel_height
                ])
    
    for track_id in dead_tracks:
        if track_id in tracked_objects_cache: 
            del tracked_objects_cache[track_id]
        if track_id in distance_history:
            del distance_history[track_id]
    
    new_frame_time = time.time()
    if prev_frame_time > 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        fps_to_display = fps 
    prev_frame_time = new_frame_time
    
    fps_text = f"FPS: {fps_to_display:.1f}"
    cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    draw_buttons(frame)
    cv2.imshow(WINDOW_NAME, frame)

# --- 6. cleanup ---
cap.release()
cv2.destroyAllWindows()
if csv_writer is not None:
    csv_file.close()
    print(f"Detection log saved to {csv_file_name}")
print("Processing finished.")