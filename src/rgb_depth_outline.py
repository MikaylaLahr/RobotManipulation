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
        depth_image = np.asanyarray(depth_frame.get_data())

        # Color Masking
        hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV) # Swap to hue, sat, values
        
        #red masking
        low_mask = np.array([0, 120, 80])
        hi_mask = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_img, low_mask, hi_mask)

        low_mask = np.array([170, 120, 80])
        hi_mask = np.array([180, 255,255])
        mask2 = cv2.inRange(hsv_img, low_mask, hi_mask)
        red_mask = mask1+mask2

        red_mask_out = color_image.copy()
        red_mask_out[np.where(red_mask==0)] = 0

        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #Brown box masking
        low_mask = np.array([5, 120, 40])
        hi_mask = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv_img, low_mask, hi_mask)
        brown_mask_out = color_image.copy()
        brown_mask_out[np.where(brown_mask==0)] = 0

        contours_brown, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_brown:
            if cv2.contourArea(contour) > 100:
                x,y,w,h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h //2
                #cv2.rectangle(color_image,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.polylines(color_image, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(color_image,(center_x,center_y),5,(0,0,255),-1)

        for contour in contours_red:
            if cv2.contourArea(contour) > 100:
                x,y,w,h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h //2
                #cv2.rectangle(color_image,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.polylines(color_image, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(color_image,(center_x,center_y),5,(0,0,255),-1)



        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET)

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

        # Find contours to detect objects based on depth
        #thresh = cv2.inRange(depth_image, lower_threshold, upper_threshold)
        ret, thresh = cv2.threshold(depth_image, 700, 255, cv2.THRESH_BINARY_INV)  # Adjust threshold
        contours_depth, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes_areas = []
        # Draw bounding boxes around detected objects
        for contour_depth in contours_depth:
            if cv2.contourArea(contour_depth) > 500:  # Filter small contours
                # Draw the contour outline
                cv2.polylines(depth_colormap, [contour_depth], isClosed=True, color=(0, 0, 0), thickness=2)

                # Optionally, calculate and display the area
                area = cv2.contourArea(contour_depth)
                bounding_boxes_areas.append(area)  # Add area to the list
                cv2.putText(depth_colormap, f"Area: {int(area)}", (contour_depth[0][0][0], contour_depth[0][0][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
        # Display the live outline image

        cv2.imshow("Live Object Outlines", outline_image)
        cv2.imshow("Live Object Outlines over RGB Image", overlay_image)
        cv2.imshow("Depth Image with Bounding Boxes", depth_colormap)
        cv2.imshow("Red Color mask out", brown_mask_out)
        cv2.imshow("Brown Mask with Center",color_image)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop the RealSense pipeline and close OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()
