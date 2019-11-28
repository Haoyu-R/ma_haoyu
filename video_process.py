import numpy as np
import cv2

cap = cv2.VideoCapture(r'..\MDM data process\video\20191029_130130_3015.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(r'..\MDM data process\video\processed_video\output.avi', fourcc, 15.0, (640, 360))

record = True
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0
while(cap.isOpened()):
    current_frame+=1
    if current_frame % 1000 == 0:
        print("Processed {:.2f} %".format(current_frame/frame_count*100))
    ret, frame = cap.read()
    if record:
        if ret:

            out.write(cv2.resize(frame, (640, 360)))
            # cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    record = not record

cap.release()
out.release()
cv2.destroyAllWindows()
