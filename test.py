# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
#
# # recognizer = cv2.face.LBPHFaceRecognizer_create()
# filename = './20191029_130130_3015.mp4'
# cap = cv2.VideoCapture(filename)
#
# frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# fig, ax = plt.subplots(1,1)
# plt.ion()
# plt.show()
#
# #Setup a dummy path
# x = np.linspace(0,width,frames)
# y = x/2. + 100*np.sin(2.*np.pi*x/1200)
#
# for i in range(frames):
#     print(1)
#     fig.clf()
#     flag, frame = cap.read()
#
#     plt.imshow(frame)
#     plt.plot(x,y,'k-', lw=2)
#     plt.plot(x[i],y[i],'or')
#
#     if cv2.waitKey(1) == 27:
#         break

import numpy as np

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])

a = a[3:9,-1,0]
print(a)