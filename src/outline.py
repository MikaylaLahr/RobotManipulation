import cv2
import pyrealsense2 as rs
import numpy as np

# Initialize the RealSense pipeline
pipeline = rs.pipeline()

# Configure the stream to capture color and depth data
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

try:
    while True:
        # Wait for a new frame
        frames = pipeline.wait_for_frames()
        
        # Get the color and depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Convert color frame to numpy array (OpenCV format)
        color_image = np.asanyarray(color_frame.get_data())

        # Convert the frame to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Optional: Find contours and draw them on the original image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the contours on a blank image
        outline_image = np.zeros_like(color_image)
        cv2.drawContours(outline_image, contours, -1, (0, 255, 0), 2)  # Green contours with thickness 2

        # Draw the contours over the original color image
        overlay_image = color_image.copy()
        cv2.drawContours(overlay_image, contours, -1, (0, 255, 0), 2)  # Green contours with thickness 2


        # Loop over each contour to find area and draw contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter out small contours (you can adjust the threshold)
                # Draw the contour on the original image
                cv2.drawContours(overlay_image, [contour], -1, (0, 255, 0), 2)  # Green contours

                # Calculate the moments of the contour to find the center of the contour for text placement
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Display the area of the contour on the image
                    cv2.putText(overlay_image, f"Area: {int(area)}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the live outline image
        cv2.imshow("Live Object Outlines", outline_image)
        cv2.imshow("Live Object Outlines over RGB Image", overlay_image)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop the RealSense pipeline and close OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()
