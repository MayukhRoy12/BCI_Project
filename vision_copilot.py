import cv2
from ultralytics import YOLO

print("Loading AI Vision Model...")
model = YOLO('yolov8n.pt') 

grasp_strategy = {
    'cup': 'Cylindrical',
    'bottle': 'Cylindrical',
    'cell phone': 'Pinch',
    'book': 'Flat',
    'apple': 'Spherical'
}


KNOWN_WIDTHS = {
    'cell phone': 7.5,
    'bottle': 7.0,      
    'cup': 8.5,         
    'book': 15.0,       
    'apple': 7.5        
}


FOCAL_LENGTH = 650 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam active. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            object_name = model.names[class_id]
            
            grasp = grasp_strategy.get(object_name, "Unknown")
            
            
            
            if object_name in KNOWN_WIDTHS:
                
                real_width = KNOWN_WIDTHS[object_name]
                pixel_width = x2 - x1 
                
                if pixel_width > 0:
                    
                    distance_cm = (real_width * FOCAL_LENGTH) / pixel_width
                    
                    label = f"{object_name.upper()} | {grasp} | Dist: {distance_cm:.1f} cm"
                    color = (0, 255, 255) # Yellow for objects with distance
                else:
                    label = f"{object_name.upper()} | {grasp}"
                    color = (0, 255, 0)
            else:
                label = f"{object_name.upper()} | {grasp}"
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("AI Copilot Vision - Multi-Object Depth", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()