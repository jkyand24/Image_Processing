import cv2
import numpy as np

kalman = cv2.KalmanFilter(4, 2) # (Dimensionality of the state, Dimensionality of the measurement)
"""
kalman.measurementMatrix: 상태 벡터와 측정 벡터 간의 선형 관계 나타냄, 측정 벡터를 상태 벡터로 변환
 [[0. 0. 0. 0.]
 [0. 0. 0. 0.]]
 
   측정 벡터: 실제로 측정되는 관측값
 
 kalman.transitionMatrix: 현재 상태 벡터를 다음 상태 벡터로 변환하는 선형 관계 나타냄, 현재 상태 벡터를 다음 상태 벡터로 예측하기 위해 사용
 [[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
 
 kalman.processNoiseCov:
 [[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
 
 kalman.statePre:
 [[0.]
 [0.]
 [0.]
 [0.]]
"""

kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.05

cap = cv2.VideoCapture('./data/slow_traffic_small.mp4')

ret, frame = cap.read()

bbox = cv2.selectROI('Select object', frame, False, False)

kalman.statePre = np.array([[bbox[0]], # 선택한 객체의 왼쪽 상단 x 좌표
                            [bbox[1]], # 선택한 객체의 왼쪽 상단 y 좌표
                            [0],
                            [0]], np.float32) # 속도 요소를 0으로 초기화 -> 초기 상태의 객체 속도 = 0 가정

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # correct: Updates the predicted state from the measurement

    kalman.correct(np.array([[np.float32(bbox[0] + bbox[2] / 2)],
                             [np.float32(bbox[1] + bbox[3] / 2)]]))
      
    # predict
      
    kalman.predict()
    
    predict_bbox = tuple(map(int, kalman.statePost[:2, 0]))
    
    cv2.rectangle(frame, 
                  (predict_bbox[0] - bbox[2] // 2, predict_bbox[1] - bbox[3] // 2),
                  (predict_bbox[0] + bbox[2] // 2, predict_bbox[1] + bbox[3] // 2),
                  (0, 255, 0),
                  2)
    
    cv2.imshow("Kalman filter tracking", frame)
    
    # 종료 조건
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()