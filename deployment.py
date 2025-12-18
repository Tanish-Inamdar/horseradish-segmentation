import cv2
import time
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "/home/tanishi2/ag group/horseradish-segmentation/YOLO_Distilled_Training/yolov8m_from_dinov3_teacher200epochs/weights/best.pt"
WEED_CLASS_ID = 1  # Verify this in your YAML (usually 0=crop, 1=weed or vice versa)

# Simulating a camera stream (Use 0 for webcam, or a video file path)
# VIDEO_SOURCE = 0 
VIDEO_SOURCE = "/home/tanishi2/ag group/dataset/field_video_test.mp4" # Update this if you have a video

# "Kill Zone" - If a weed center enters this box, we spray.
# Format: [x_start_percent, y_start_percent, x_end_percent, y_end_percent]
# This defines a vertical strip in the middle of the image
SPRAY_ZONE_PCT = [0.4, 0.0, 0.6, 1.0] 

def main():
    model = YOLO(MODEL_PATH)
    
    # Open video stream
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Cannot open video source: {VIDEO_SOURCE}")
        print("ℹ️ Try setting VIDEO_SOURCE = 0 for webcam.")
        return

    print("--- 🚜 Starting Virtual Sprayer System ---")
    print("Press 'q' to quit.")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        
        # Calculate Spray Zone coordinates (Pixel values)
        z_x1 = int(width * SPRAY_ZONE_PCT[0])
        z_y1 = int(height * SPRAY_ZONE_PCT[1])
        z_x2 = int(width * SPRAY_ZONE_PCT[2])
        z_y2 = int(height * SPRAY_ZONE_PCT[3])

        # Run Inference
        results = model(frame, verbose=False, conf=0.4) # conf=0.4 filters weak detections
        
        # Annotate frame with the "Kill Zone"
        cv2.rectangle(frame, (z_x1, z_y1), (z_x2, z_y2), (255, 0, 0), 2)
        cv2.putText(frame, "SPRAY ZONE", (z_x1, z_y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        spray_triggered = False

        # Process Detections
        for r in results:
            if r.masks is None: continue
            
            # Get boxes and classes
            boxes = r.boxes
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                
                # Check if it is a WEED
                if cls_id == WEED_CLASS_ID:
                    # Get the mask for this specific object
                    # r.masks.xy gives coordinates of the mask contour
                    mask_contour = r.masks.xy[i]
                    
                    if len(mask_contour) > 0:
                        # Calculate Centroid using Moments
                        M = cv2.moments(mask_contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            
                            # Draw the weed centroid
                            cv2.circle(frame, (cX, cY), 7, (0, 0, 255), -1)

                            # --- LOGIC: IS IT IN THE ZONE? ---
                            if z_x1 < cX < z_x2 and z_y1 < cY < z_y2:
                                spray_triggered = True
                                # Visual indicator of "Spray"
                                cv2.circle(frame, (cX, cY), 20, (0, 255, 255), 3)
                                cv2.putText(frame, "TARGET ACQUIRED", (cX + 10, cY), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Status Display
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if spray_triggered:
            cv2.putText(frame, "!!! SPRAYING !!!", (width//2 - 100, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # HERE is where you would call: GPIO.output(PIN, HIGH)

        # Show the feed
        cv2.imshow('Robot View', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()