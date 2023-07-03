import cv2

# 비디오에서 첫 frame을 가져와, ROI를 직접 설정하기

cap = cv2.VideoCapture('./data/slow_traffic_small.mp4')

ret, frame = cap.read()
x, y, w, h = cv2.selectROI("Select Object", frame, showCrosshair = False, fromCenter = False)

roi = frame[y:y+h, x:x+w]

cv2.imshow('roi test', roi)
cv2.waitKey(0)

# ROI의 distribution 확인하기

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist(images = [hsv_roi],
                        channels = [0],
                        mask = None, # 이미지의 분석 영역, None은 전체 영역 
                        histSize = [180], # 각 차원의 bin 개수
                        ranges = [0, 180]) # The first and second elements of each pair specify the lower and upper boundaries.

cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

#

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

track_window = (x, y, w, h)

while True:
    # 새 frame을 가져와 HSV로 전환
    
    ret, frame = cap.read()
    
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 히스토그램 역투영: 영상의 각 픽셀이 주어진 히스토그램 모델에 얼마나 일치하는지 검사 -> 임의의 색상 영역을 검출
    
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    
    # meanshift하여 frame에 표시
    """ 
    meanshift: 원형 윈도우가 있고, 이 윈도우를 최대 픽셀 밀도값 (또는 포인트의 최대 개수)을 가지는 영역으로 이동시킴.
    
    원의 중심을 새롭게 구한 무게중심점으로 이동시키기를 반복 - 원의 중심과 윈에 포함된 포인트의 무게중심점 위치가 동일할 때까지 
    => 최종 원형 윈도우는 최대 픽셀 분포를 가지게 됨 
    """
    
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    
    x, y, w, h = track_window
    print(x, y, w, h)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Mean shift tracking", frame)
    
    # 종료 조건
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()