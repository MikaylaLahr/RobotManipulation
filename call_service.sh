rosservice call /generate_packing_plan "detections:
  header:
    seq: 0
    stamp: now
    frame_id: 'map'
  detections:
    - header:
        seq: 0
        stamp: now
        frame_id: 'map'
      results: # This is an array of vision_msgs/ObjectHypothesisWithPose
        - id: 0
          score: 0.0
          pose: # geometry_msgs/PoseWithCovariance - needed for ObjectHypothesisWithPose
            pose:
              position: {x: 0.0, y: 0.0, z: 0.0}
              orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
            covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      bbox: # vision_msgs/BoundingBox3D - needed for Detection3D
        center:
          position: {x: 1.0, y: 2.0, z: 0.5}
          orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
        size: {x: 0.5, y: 0.5, z: 1.0} # Dummy dimensions
"
