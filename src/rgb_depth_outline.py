import cv2
import pyrealsense2 as rs
import numpy as np

# Initialize the RealSense pipeline
pipeline = rs.pipeline()

# Configure the stream to capture color and depth data
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

align = rs.align(rs.stream.color)

# Start the pipeline
pipeline.start(config)

try:
    while True:
        # Wait for a new frame
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        
        # Get the color and depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # Convert color frame to numpy array (OpenCV format)
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Color Masking
        hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV) # Swap to hue, sat, values

        depth_mask = cv2.inRange(depth_image, 600, 700)  # Adjust threshold
        
        # Red box masking
        low_mask = np.array([0, 140, 20])
        hi_mask = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_img, low_mask, hi_mask)

        low_mask = np.array([170, 140, 20])
        hi_mask = np.array([180, 255,255])
        mask2 = cv2.inRange(hsv_img, low_mask, hi_mask)
        red_mask = (mask1 | mask2) & depth_mask
        red_mask_out = color_image.copy()
        red_mask_out[np.where(red_mask==0)] = 0

        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        red_area = np.count_nonzero(red_mask)
        print(red_area)

        # Black cube masking
        low_mask = np.array([0, 0, 0])
        hi_mask = np.array([180, 140, 100])
        black_mask = cv2.inRange(hsv_img, low_mask, hi_mask)
        kernel = np.ones((5, 5), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
        black_mask = black_mask & depth_mask
        black_mask_out = color_image.copy()
        black_mask_out[black_mask == 0] = 0

        contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Blue cube masking
        low_mask = np.array([100, 150, 50])
        hi_mask = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv_img, low_mask, hi_mask)
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = blue_mask & depth_mask
        blue_mask_out = color_image.copy()
        blue_mask_out[blue_mask == 0] = 0

        contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # Convert the frame to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Optional: Find contours and draw them on the original image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours over the original color image
        general_contours_overlay = color_image.copy()
        cv2.drawContours(general_contours_overlay, contours, -1, (0, 255, 0), 2)  # Green contours with thickness 2
        
        # Draw the contours on a blank image
        outline_only_image = np.zeros_like(color_image)
        cv2.drawContours(outline_only_image, contours, -1, (0, 255, 0), 2)  # Green contours with thickness 2

        masked_contour_overlay = color_image.copy()

        for contour in contours_black:
            if cv2.contourArea(contour) > 500:
                x,y,w,h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.polylines(masked_contour_overlay, [contour], isClosed=True, color=(0, 255, 255), thickness=2)
                cv2.circle(masked_contour_overlay,(center_x,center_y),5,(0,255,255),-1)

        for contour in contours_red:
            if cv2.contourArea(contour) > 500:
                x,y,w,h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.polylines(masked_contour_overlay, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(masked_contour_overlay,(center_x,center_y),5,(0,255,0),-1)

        for contour in contours_blue:
            if cv2.contourArea(contour) > 500:
                x,y,w,h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.polylines(masked_contour_overlay, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(masked_contour_overlay,(center_x,center_y),5,(0,255,0),-1)

        # Loop over each contour to find area and draw contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter out small contours (you can adjust the threshold)
                # Draw the contour on the original image
                cv2.drawContours(general_contours_overlay, [contour], -1, (0, 255, 0), 2)  # Green contours

                # Calculate the moments of the contour to find the center of the contour for text placement
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Display the area of the contour on the image
                    cv2.putText(general_contours_overlay, f"Area: {int(area)}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Find contours to detect objects based on depth
        #thresh = cv2.inRange(depth_image, lower_threshold, upper_threshold)
        contours_depth, _ = cv2.findContours(depth_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET)

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
                
        cv2.imshow("General Object Outlines", outline_only_image)
        cv2.imshow("General Object Outlines over RGB Image", general_contours_overlay)
        cv2.imshow("Depth Image with General Object Outlines", depth_colormap)
        cv2.imshow("Contours found from thresholded images", masked_contour_overlay)
        cv2.imshow("Red mask", red_mask_out)
        cv2.imshow("Black cube mask", black_mask_out)
        cv2.imshow("Blue cube mask", blue_mask_out)
        cv2.imshow("Blue cube mask", red_mask_out+blue_mask_out)
        cv2.imshow("Depth mask", depth_mask)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop the RealSense pipeline and close OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()
