import cv2

cap = cv2.VideoCapture('./data/slow_traffic_small.mp4')

orb = cv2.ORB_create()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    keypoints = orb.detect(gray, None) # keypoints는 frame마다 매번 새로워짐, 한 frame 당 500개
    
    print(keypoints[0].pt, keypoints[1].pt)
    
    frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
    
    cv2.imshow("ORB", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()