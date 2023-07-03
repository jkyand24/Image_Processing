import cv2
import os

cap = cv2.VideoCapture('./data/blooms-113004.mp4')

fps = 25

count = 0

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        
        if ret:
            if (int(cap.get(1)) % fps == 0):
                os.makedirs('./data/p223_saved/', exist_ok=True)
                cv2.imwrite(
                    f"./data/p223_saved/image_{str(count).zfill(4)}.png", frame
                )
                count += 1
                
        else:
            break

else:
    print("Can't open video.")
    
cap.release()
cv2.destroyAllWindows()