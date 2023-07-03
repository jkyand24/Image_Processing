import cv2

cap = cv2.VideoCapture('./data/slow_traffic_small.mp4')

# prev_gray, prev_points

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
prev_corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
prev_points = prev_corners.squeeze()

#

lk_params = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = (0, 255, 0)

while True:
    # gray, next_points
    
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
    
    # 현재 frame에 그리기

    for i, (prev_point, next_point) in enumerate(zip(prev_points, next_points)):
        x1, y1 = prev_point.astype(int)
        x2, y2 = next_point.astype(int)
        
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (x2, y2), 3, color, -1)
        
    cv2.imshow("Feature-based Tracking", frame)
    
    # prev_gray, prev_points 업데이트

    prev_gray = gray.copy()
    prev_points = next_points

    # 종료 조건
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
#

cap.release()
cv2.destroyAllWindows()