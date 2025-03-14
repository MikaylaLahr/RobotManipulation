import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the stream for depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Loop for processing frames
while True:
    # Wait for a frame from the camera
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame:
        continue

    # Convert depth frame to numpy array
    depth_image = np.asanyarray(depth_frame.get_data())

    # Convert color frame to numpy array (RGB image)
    color_image = np.asanyarray(color_frame.get_data())

    # Normalize the depth image to the 0-255 range for visualization
    depth_image_normalized = cv2.convertScaleAbs(depth_image, alpha=0.03)

    # Display the RGB image for debugging
    cv2.imshow("RGB Image", color_image)

    # Display the normalized depth image for debugging
    cv2.imshow("Normalized Depth Image", depth_image_normalized)

    # Convert the depth image to a point cloud
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    width = depth_image.shape[1]
    height = depth_image.shape[0]
    
    # Create a point cloud
    points = []
    for y in range(height):
        for x in range(width):
            # Get the depth value at (x, y)
            depth = depth_image[y, x]

            if depth == 0:  # Ignore invalid depth points (depth = 0)
                continue

            # Convert depth to 3D point using the camera intrinsics
            point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
            points.append(point)

    # Convert the list of 3D points to a numpy array
    points = np.array(points)

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud (optional)
    #o3d.visualization.draw_geometries([pcd])

    # Now, project the 3D points into a 2D plane (XY plane) for contour detection
    # This will give us the 2D projection of the points.
    points_2d = points[:, :2]  # Take the x and y coordinates of the 3D points

    # Normalize points to be in the image plane
    points_2d -= np.min(points_2d, axis=0)  # Translate points to positive space
    points_2d /= np.max(points_2d, axis=0)  # Scale points to the range [0, 1]
    points_2d = (points_2d * 255).astype(np.uint8)  # Scale to [0, 255] for visualization

    # Create an empty black image for contour detection
    contour_image = np.zeros((480, 640), dtype=np.uint8)

    # Draw the points as white pixels on the black canvas (image)
    for point in points_2d:
        x, y = point
        contour_image[y, x] = 255

    # Find contours in the 2D binary image
    contours, _ = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image (or on the point cloud 2D projection)
    contour_image_colored = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image_colored, contours, -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Contours on Point Cloud", contour_image_colored)

    # Wait for user input to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Stop streaming
pipeline.stop()
cv2.destroyAllWindows()
