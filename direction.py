import cv2
from ultralytics import YOLO


# Load the YOLOv8n model
model = YOLO('yolov8n.pt')  # Make sure the path to your model is correct

# Open video file
cap = cv2.VideoCapture('C:\\Users\\lpr-direction\\sample_new.mp4')  # Make sure the video path is correct

# Dictionary to track vehicle centroids
vehicle_tracks = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or can't read the video.")
        break

    # Get predictions
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        confidences = result.boxes.conf  # Confidence scores
        class_ids = result.boxes.cls  # Class IDs

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            if class_id in [2, 3, 5, 7]:  # Filter for vehicle classes (car, truck, bus, motorcycle)
                x1, y1, x2, y2 = map(int, box)
                # Calculate centroid
                x_centroid = (x1 + x2) // 2
                y_centroid = (y1 + y2) // 2
                centroid = (x_centroid, y_centroid)

                # Track vehicle ID (simple ID assignment)
                vehicle_id = len(vehicle_tracks)
                if vehicle_id not in vehicle_tracks:
                    vehicle_tracks[vehicle_id] = []

                # Store current centroid for this vehicle
                vehicle_tracks[vehicle_id].append(centroid)

                # Determine direction based on the last centroid
                if len(vehicle_tracks[vehicle_id]) > 1:
                    previous_centroid = vehicle_tracks[vehicle_id][-2]
                    if y_centroid > previous_centroid[1]:
                        direction = "Down"
                        color = (0, 0, 255)  # Red for down
                        # Draw an arrow pointing down
                        cv2.arrowedLine(frame, (x_centroid, y_centroid), (x_centroid, y_centroid + 20), color, 2, tipLength=0.2)
                    else:
                        direction = "Up"
                        color = (0, 255, 0)  # Green for up
                        # Draw an arrow pointing up
                        cv2.arrowedLine(frame, (x_centroid, y_centroid), (x_centroid, y_centroid - 20), color, 2, tipLength=0.2)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw direction text
                    cv2.putText(frame, direction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show frame
    cv2.imshow('Vehicle Direction Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
