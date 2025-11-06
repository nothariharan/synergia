import cv2

# --- Try to open the camera ---
# We will try index 0, 1, and 2
# We will try DSHOW and the default driver
#
# !! Make sure ALL other apps using your camera are CLOSED !!
#
CAM_INDEX = 0  # <--- Start with 0

cap = cv2.VideoCapture(CAM_INDEX + cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Error: Could not open camera at index {CAM_INDEX} with DSHOW.")
    print("--- Trying default driver... ---")
    cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open camera at index {CAM_INDEX} with ANY driver.")
    print("\n--- TROUBLESHOOTING ---")
    print("1. Did you check Windows Camera Privacy Settings?")
    print("2. Did you disable virtual cameras (OBS, etc.) in Device Manager and restart?")
    print("3. Try changing CAM_INDEX to 1 or 2 in this script.")
    exit()

print("\n--- SUCCESS! Camera is open. ---")
print("Press 'q' to quit.")

# Set properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to grab frame.")
        break
        
    cv2.imshow("Test Camera", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()