import cv2

def find_available_cameras():
    print("Scanning for available cameras...")
    max_cameras_to_check = 10
    
    for index in range(max_cameras_to_check):
        print(f"--- Checking index {index} ---")
        
        # --- Try with the default driver first ---
        cap_default = cv2.VideoCapture(index)
        if cap_default.isOpened():
            print(f"SUCCESS: Camera found at index {index} (Default Driver)")
            cap_default.release()
            continue # Move to the next index
        
        # --- If default fails, try with DSHOW ---
        cap_dshow = cv2.VideoCapture(index + cv2.CAP_DSHOW)
        if cap_dshow.isOpened():
            print(f"SUCCESS: Camera found at index {index} (DSHOW Driver)")
            cap_dshow.release()
            continue
            
        # --- If DSHOW fails, try with MSMF ---
        cap_msmf = cv2.VideoCapture(index + cv2.CAP_MSMF)
        if cap_msmf.isOpened():
            print(f"SUCCESS: Camera found at index {index} (MSMF Driver)")
            cap_msmf.release()
            continue
            
        print(f"Index {index} is not available with any driver.")

    print("--- Scan complete. ---")

if __name__ == "__main__":
    find_available_cameras()