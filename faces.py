import cv2
import numpy as np
import time

# Load the model
net = cv2.dnn.readNet('face-detection-retail-0004.xml',
                      'face-detection-retail-0004.bin')

net2 = cv2.dnn.readNet('emotions-recognition-retail-0003.xml',
                       'emotions-recognition-retail-0003.bin')

# Specify target device
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net2.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

cap = cv2.VideoCapture('my_face.avi')

emo =  ('neutral', 'happy', 'sad', 'surprise', 'anger')

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (50,200)
fontScale              = 4
fontColor              = (255,0,255)
lineType               = 3


count = 0
while(cap.isOpened()):
    try:
        ret, frame = cap.read()
    except:
        break

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()
    out = out[0,0,:,:]
    frame = resized_image = cv2.resize(frame, (int(frame.shape[1]/1), int(frame.shape[0]/1))) 

    hasFace = False

    for detection in out.reshape(-1, 7): 
        confidence = float(detection[2]) 
        xmin = int(detection[3] * frame.shape[1]) 
        ymin = int(detection[4] * frame.shape[0]) 
        xmax = int(detection[5] * frame.shape[1]) 
        ymax = int(detection[6] * frame.shape[0])
        if confidence > 0.5:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 153, 55), 2)
            face_img = frame[ymin:ymax, xmin:xmax]
            hasFace = True
 
    if hasFace:
        blob2 = cv2.dnn.blobFromImage(frame, size=(64, 64), swapRB=False)
        net2.setInput(blob2)
        out2 = net2.forward()
        out2 = out2[0,:,0,0]
        if out2[np.argmax(out2)] > 0.1:
            cv2.putText(frame,emo[np.argmax(out2)], 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)
        else:
            print("-")

    cv2.imwrite("frame%d.jpg" % count, frame) 
    count += 1
    print(count)
    if count>600:
        print(g)
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
