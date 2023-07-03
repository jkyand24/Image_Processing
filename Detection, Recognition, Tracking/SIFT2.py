import cv2

cap = cv2.VideoCapture('./data/slow_traffic_small.mp4')

sift = cv2.SIFT_create(contrastThreshold=0.02)

max_keypoints = 100

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    if len(keypoints) > max_keypoints:
        keypoints = sorted(keypoints, key = lambda x: -x.response)[:max_keypoints]
    
    frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow("SIFT", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()