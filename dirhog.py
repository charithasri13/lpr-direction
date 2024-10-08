import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model for vehicle detection (COCO model)
model = YOLO('yolov8n.pt')

# Initialize a dictionary to keep track of vehicle positions and sizes
vehicle_data = {}

# Load the image
image_path = 'C:\\Users\\chari\\OneDrive\\Desktop\\dir4.jpg'
image = cv2.imread(image_path)

# Detect vehicles in the image using YOLOv8
results = model(image)
detections = []

# Process the YOLOv8 detections and prepare vehicle tracking data
for result in results[0].boxes.data:
    x1, y1, x2, y2, confidence, class_id = result[:6].cpu().numpy()

    # Check if detected class is a vehicle (e.g., cars, motorcycles, buses, trucks)
    if int(class_id) in [2, 3, 5, 7]:  # COCO vehicle classes: car, motorcycle, bus, truck
        detections.append([x1, y1, x2, y2, confidence])

# Loop through the detected vehicles
for i, detection in enumerate(detections):
    x1, y1, x2, y2, confidence = map(int, detection[:5])

    # Calculate the bounding box width and height
    width = x2 - x1
    height = y2 - y1

    # Extract the detected vehicle region
    vehicle_region = image[y1:y2, x1:x2]

    # Resize the vehicle region to a fixed size (e.g., 128x128) for HOG feature extraction
    fixed_size = (128, 128)
    resized_vehicle = cv2.resize(vehicle_region, fixed_size)

    # Convert the resized vehicle region to grayscale for feature detection
    gray_vehicle = cv2.cvtColor(resized_vehicle, cv2.COLOR_BGR2GRAY)

    # Use Histogram of Oriented Gradients (HOG) to extract features from the resized vehicle region
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray_vehicle)

    # Placeholder logic for determining direction based on orientation
    # Here, we use a heuristic approach to determine if the vehicle is facing forward (front) or backward (rear)
    # For demonstration, we classify the vehicle using HOG feature analysis and apply simple threshold logic

    front_view_threshold = 0.5  # This threshold should be adjusted based on training data
    average_hog_feature = np.mean(hog_features)

    # Logic to determine if the vehicle is coming closer ("in") or going away ("out")
    if average_hog_feature < front_view_threshold:
        direction = "In"  # Front view (coming towards)
    else:
        direction = "Out"  # Rear view (going away)

    # Display the ID and direction on the image
    vehicle_id = i + 1  # Assign an ID to each vehicle (starting from 1)
    cv2.putText(image, f"ID: {vehicle_id}, Direction: {direction}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

# Display the image with IDs and directions
cv2.imshow('Vehicle Detection and Direction', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
