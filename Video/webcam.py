import cv2

cap = cv2.VideoCapture(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"Original width and height: {width} * {height}")

while True:
    ret, frame = cap.read()
    
    frame = cv2.resize(frame, (864, 486))
    
    cv2.imshow('Webcam', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()