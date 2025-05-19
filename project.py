!pip install opencv-python
!pip install pillow

# Traffic Light Timing System
# This code detects vehicles in images and calculates appropriate green light timing

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import io
from IPython.display import display, HTML, clear_output

# Install required dependencies
!pip install -q torch torchvision
!pip install -q opencv-python-headless
!pip install -q matplotlib
!pip install -q Pillow

# Clone YOLOv5 repository (lightweight version)
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -q -r requirements.txt

# Load YOLOv5 model (using small version for better performance)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Vehicle classes in COCO dataset
vehicle_classes = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']

def detect_vehicles(image_path):
    """
    Detect vehicles in an image using YOLOv5.

    Args:
        image_path: Path to the image file

    Returns:
        detected_vehicles: Dictionary containing vehicle counts by type
        annotated_image: Image with bounding boxes around detected vehicles
    """
    # Load image
    img = Image.open(image_path)

    # Perform inference
    results = model(img)

    # Extract predictions
    predictions = results.pandas().xyxy[0]

    # Filter for vehicle classes
    vehicles = predictions[predictions['name'].isin(vehicle_classes)]

    # Count vehicles by type
    vehicle_counts = {}
    for vehicle_type in vehicle_classes:
        count = len(vehicles[vehicles['name'] == vehicle_type])
        if count > 0:
            vehicle_counts[vehicle_type] = count

    # Calculate total
    total_vehicles = sum(vehicle_counts.values())
    vehicle_counts['total'] = total_vehicles

    # Create annotated image
    annotated_img = results.render()[0]

    return vehicle_counts, annotated_img

def calculate_green_light_time(vehicle_counts, base_time=30, vehicle_weight=2):
    """
    Calculate appropriate green light time based on vehicle counts.

    Args:
        vehicle_counts: Dictionary containing vehicle counts by type
        base_time: Minimum green light time in seconds
        vehicle_weight: Additional seconds per vehicle

    Returns:
        green_light_time: Calculated green light time in seconds
    """
    # Different vehicle types can have different weights
    type_weights = {
        'car': 1.0,
        'motorcycle': 0.8,
        'bicycle': 0.5,
        'bus': 2.5,
        'truck': 2.0
    }

    # Calculate weighted sum
    weighted_count = 0
    for vehicle_type, count in vehicle_counts.items():
        if vehicle_type in type_weights and vehicle_type != 'total':
            weighted_count += count * type_weights[vehicle_type]

    # Calculate green light time (base time + additional time per weighted vehicle)
    green_light_time = base_time + weighted_count * vehicle_weight

    # Cap the maximum time to prevent overly long green lights
    max_time = 90
    green_light_time = min(green_light_time, max_time)

    return int(green_light_time)

def traffic_light_simulation(green_time):
    """
    Simple visual simulation of a traffic light.

    Args:
        green_time: Time for the green light in seconds
    """
    colors = ['red', 'yellow', 'green']
    times = [5, 2, green_time]  # Red: 5s, Yellow: 2s, Green: calculated time

    for color, t in zip(colors, times):
        clear_output(wait=True)
        display(HTML(f"""
        
            
            
            
        
        Current signal: {color.upper()}
        Time remaining: {t}s
        """))

        # Simulate countdown
        for remaining in range(t-1, -1, -1):
            time.sleep(1)
            clear_output(wait=True)
            display(HTML(f"""
            
                
                
                
            
            Current signal: {color.upper()}
            Time remaining: {remaining}s
            """))

def process_image_and_calculate_timing(image_path):
    """
    Process an image to detect vehicles and calculate appropriate green light time.

    Args:
        image_path: Path to the image file
    """
    # Detect vehicles
    print("Detecting vehicles...")
    vehicle_counts, annotated_img = detect_vehicles(image_path)

    # Display original image
    print("\nOriginal Image:")
    img = Image.open(image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(np.array(img))
    plt.axis('off')
    plt.show()

    # Display annotated image
    print("\nDetected Vehicles:")
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Display vehicle counts
    print("\nVehicle Counts:")
    for vehicle_type, count in vehicle_counts.items():
        if vehicle_type != 'total':
            print(f"{vehicle_type.capitalize()}: {count}")
    print(f"Total Vehicles: {vehicle_counts.get('total', 0)}")

    # Calculate and display green light time
    green_time = calculate_green_light_time(vehicle_counts)
    print(f"\nCalculated Green Light Time: {green_time} seconds")

    # Ask if user wants to see the simulation
    simulate = input("\nDo you want to see the traffic light simulation? (y/n): ")
    if simulate.lower() == 'y':
        traffic_light_simulation(green_time)

# Example usage with upload functionality
from google.colab import files
print("Please upload an image with traffic...")
uploaded = files.upload()

# Process the uploaded image
for filename in uploaded.keys():
    print(f"\nProcessing image: {filename}")
    process_image_and_calculate_timing(filename)
     
