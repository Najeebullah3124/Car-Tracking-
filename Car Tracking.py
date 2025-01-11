from ultralytics import YOLO
import cv2
from sort import Sort
import numpy as np

# Load the YOLO model and explicitly set the device to CUDA
model = YOLO("yolov8l.pt").to("cuda")  # Replace with your model path

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Open the video file
cap = cv2.VideoCapture("C:/Users/Najeeb ULLAH/Desktop/CarTest.mp4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define the counting line (y-coordinate)
line_position = 400
car_count = 0

# Store IDs of cars that have been counted
counted_ids = set()

while True:
    success, img = cap.read()
    if not success:
        print("End of video or error in reading frame.")
        break

    # Perform object detection
    results = model(img, stream=True)

    detections = np.empty((0, 5))  # Initialize an empty array for detections

    # Loop over detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates and confidence, converting tensors to CPU and then NumPy
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
            conf = box.conf[0].cpu().numpy()  # Confidence score
            cls = int(box.cls[0].cpu().numpy())  # Class ID (as integer)

            # Check if the detected object is a car
            if model.names[cls] == 'car' and conf > 0.5:  # Filter low-confidence detections
                # Add detection to the array
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))  # Stack detections

                # Draw bounding box and label on the frame
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = f"Car: {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Update the SORT tracker
    tracked_objects = tracker.update(detections)

    # Draw the counting line
    cv2.line(img, (0, line_position), (img.shape[1], line_position), (0, 255, 0), 2)

    # Loop over tracked objects
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)  # Extract bounding box and tracking ID
        center_y = (y1 + y2) // 2  # Calculate the vertical center of the bounding box

        # Check if the car crosses the counting line
        if track_id not in counted_ids and center_y > line_position - 5 and center_y < line_position + 5:
            car_count += 1
            counted_ids.add(track_id)

        # Draw tracking ID and bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for tracked objects
        cv2.putText(img, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the car count on the frame
    cv2.putText(img, f"Car Count: {car_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Car Counter", img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the video capture
cv2.destroyAllWindows()  # Close all OpenCV windows
