import cv2
import numpy as np

cap = cv2.VideoCapture('./data/slow_traffic_small.mp4')

orb = cv2.ORB_create()

min_keypoint_size = 10
duplicate_threshold = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    keypoints = orb.detect(gray, None) # keypoints는 frame마다 매번 새로워짐, 한 frame 당 500개
    
    keypoints = [kp for kp in keypoints if kp.size > min_keypoint_size]
    
    mask = np.ones(len(keypoints), dtype=bool)
    for i, kp1 in enumerate(keypoints):
        if mask[i]:
            for j, kp2 in enumerate(keypoints[i+1:]):
                if mask[i+j+1] \
                and np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt)) < duplicate_threshold:
                    mask[i+j+1] = False
    
    keypoints = [kp for i, kp in enumerate(keypoints) if mask[i]]
    
    frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
    
    cv2.imshow("ORB", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()