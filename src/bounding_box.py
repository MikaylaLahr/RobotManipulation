import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the stream for depth
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# List to store bounding box areas
bounding_boxes_areas = []


# Loop for processing frames
while True:
    # Wait for a frame from the camera
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    if not depth_frame:
        continue

    # Convert depth frame to numpy array
    depth_image = np.asanyarray(depth_frame.get_data())

    print("Depth Image Min Value:", np.min(depth_image))
    print("Depth Image Max Value:", np.max(depth_image))

    # Normalize the depth image to the 0-255 range for visualization
    depth_image_normalized = cv2.convertScaleAbs(depth_image, alpha=0.03)

    # Display the normalized depth image for debugging
    cv2.imshow("Normalized Depth Image", depth_image_normalized)

    # Normalize the depth image to the 0-255 range for visualization
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET)

    # Set threshold range based on your min/max depth values
    lower_threshold = 500   # Set this based on the minimum distance you want to detect (in depth units)
    upper_threshold = 3000  # Set this based on the maximum distance you want to detect

    # Find contours to detect objects based on depth
    #thresh = cv2.inRange(depth_image, lower_threshold, upper_threshold)
    ret, thresh = cv2.threshold(depth_image, 700, 255, cv2.THRESH_BINARY_INV)  # Adjust threshold
    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected objects
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            # Draw the contour outline
            cv2.polylines(depth_colormap, [contour], isClosed=True, color=(0, 0, 0), thickness=2)

            # Optionally, calculate and display the area
            area = cv2.contourArea(contour)
            bounding_boxes_areas.append(area)  # Add area to the list
            cv2.putText(depth_colormap, f"Area: {int(area)}", (contour[0][0][0], contour[0][0][1] - 10), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
    # Sort bounding boxes by area (if you want to rank them)
    sorted_areas = sorted(bounding_boxes_areas, reverse=True)  # Sort in descending order of area

    # You can display or use sorted_areas for further comparison
    #print(f"Bounding box areas (sorted): {sorted_areas}")


    # Show the depth image with bounding boxes
    cv2.imshow("Raw Depth Image", depth_image)
    cv2.imshow("Depth Image with Bounding Boxes", depth_colormap)
    cv2.imshow("Thresholded Image", thresh)  # Visualize the thresholded binary image


    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop streaming
pipeline.stop()
cv2.destroyAllWindows()
