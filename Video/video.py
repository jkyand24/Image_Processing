import cv2

cap = cv2.VideoCapture('./data/blooms-113004.mp4')

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

print(f"Original width and height: {width} * {height}")
print(f"fps: {fps}")
print(f"frame count: {frame_count}")

if cap.isOpened(): # 비디오 캡쳐가 준비되었는지 확인
    while True: # 아래를 계속 반복하기. 이게 없으면 -> SyntaxError: 'break' outside loop
        ret, frame = cap.read() # read: Grabs, decodes and returns the next video frame
        if ret:
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow('Video name', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
            # waitKey의 반환값의 뒷부분만 취하기 위해, & 0xFF를 해줌
            # ord: 하나의 문자를 인자로 받고, 그 문자에 해당하는 유니코드 정수를 반환
                exit()
        else:
            break
else:
    print("Can't open video.")
    
cap.release() # release: Closes video file or capturing device

cv2.destroyAllWindows()