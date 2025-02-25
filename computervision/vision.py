import cv2
import depthai as dai
import mediapipe as mp
import math
from utils import calculate_angle 
from utils import count_extended_fingers

# MediaPipe 
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)


# DepthAI
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)  
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)


# Main

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    
    cv2.namedWindow("Right Arm & Hand Pose", cv2.WINDOW_NORMAL)
    
    while True:
        in_rgb = q_rgb.tryGet()
        if in_rgb is None:
            continue
        frame = in_rgb.getCvFrame()  
        image_height, image_width, _ = frame.shape

        # Process MediaPipe Pose
        pose_results = pose.process(frame)

        # Process MediaPipe Hands
        hands_results = hands.process(frame)

        # copy for annotation
        annotated_frame = frame.copy()

        
        # Right shoulder: 12, Right elbow: 14, Right wrist: 16
        if pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks.landmark
            
            # coordinates:
            right_shoulder = (int(pose_landmarks[12].x * image_width), int(pose_landmarks[12].y * image_height))
            right_elbow    = (int(pose_landmarks[14].x * image_width), int(pose_landmarks[14].y * image_height))
            right_wrist    = (int(pose_landmarks[16].x * image_width), int(pose_landmarks[16].y * image_height))
            
            # circles
            cv2.circle(annotated_frame, right_shoulder, 5, (0, 255, 0), -1)
            cv2.circle(annotated_frame, right_elbow, 5, (0, 255, 0), -1)
            cv2.circle(annotated_frame, right_wrist, 5, (0, 255, 0), -1)
            
            #lines 
            cv2.line(annotated_frame, right_shoulder, right_elbow, (0, 255, 0), 2)
            cv2.line(annotated_frame, right_elbow, right_wrist, (0, 255, 0), 2)
            
            # elbow angle
            elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            cv2.putText(annotated_frame, f"Elbow Angle: {int(elbow_angle)} deg", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Process Hand landmarks
        right_hand_landmarks = None
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                label = handedness.classification[0].label  
                if label == "Right":
                    right_hand_landmarks = hand_landmarks
                   
                    mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    finger_count = count_extended_fingers(hand_landmarks, label, image_width, image_height)
                    cv2.putText(annotated_frame, f"Right Hand Fingers: {finger_count}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    break 
        
        #  add logic to determine specific arm or finger movements based on landmarks,
        # e.g., checking if the elbow is bending or if fingers are opening/closing.
        
        
        
        annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("OAK-D Right Arm & Hand Pose", annotated_bgr)
        
        if cv2.waitKey(1) == ord('q'):
            break


pose.close()
hands.close()
cv2.destroyAllWindows()
