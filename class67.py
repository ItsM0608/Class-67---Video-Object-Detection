import cv2
import numpy as np

state = "play"
 

# Path to model configuration and weights files
modelConfiguration='static/cfg/yolov3.cfg'
modelWeights='yolov3.weights'

labels=open("coco.names").read().strip().split('\n')
print(labels)

confidenceThreshold=0.5
# Load YOLO object detection network
yoloNetwork=cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)


video= cv2.VideoCapture("static/bb2.mp4")
while True:
    if (state=="play"):
        
        ret,image=video.read()
        image=cv2.resize(image,(600,600))
        #print(ret)

        dimensions = image.shape[:2]
        Height=dimensions[0]
        Width=dimensions[1]
        print(dimensions)
        NMSThreshold=0.3

        # Create blob from image and set input for YOLO network
        blob= cv2.dnn.blobFromImage(image, 1/255, (416,416))
        #print(blob)
        # Syntax: blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size)
        # 1/255 is takes to normalise the pixel value from 0-255 to 0-1 as the yolo (other models also) require the pixel to be in range 0 to 1.
        # 416,416 is size of images taken by yolo model
        # Input the image blob to hte model
        yoloNetwork.setInput(blob)
        # get names of unconnected outputlayers
        layerName = yoloNetwork.getUnconnectedOutLayersNames()
        print(layerName)

        layersOutputs=yoloNetwork.forward(layerName)

        boxes=[]
        confidences=[]
        classIds=[]

        for output in layersOutputs:
            for detection in output:
                score=detection[5:]
                classId=np.argmax(score)
                confidence=score[classId]

                if confidence >confidenceThreshold:
                    box=detection[0:4] *np.array([Width,Height,Width,Height])
                    (centerX,centerY,w,h)= box.astype('int')
                    x=int(centerX -(w/2))
                    y=int(centerY -(h/2))

                    boxes.append([x,y,int(w),int(h)])
                    confidences.append(float(confidence))
                    classIds.append(classId)

        indexes=cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold,NMSThreshold)
        for i in range(len(boxes)):
            if i in indexes:
                # x=boxes[i][0]
                # y=boxes[i][1]
                # w=boxes[i][2]
                # h=boxes[i][3]
                # cv2.rectangle(image, (x,y), (x+w,y+h),(0,0,255),2)
                # label=labels[classIds[i]]

                # text='{}:{:2f}'.format(label,confidences[i]*100)
                # cv2.putText(image,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                if labels[classIds[i]] == "sports ball":
                    x,y,w,h = boxes[i]
                    if i%2==0:
                        color=(0,255,0)

                    else:
                        color=(255,0,0)
                    cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
                    label=labels[classIds[i]]
                    cv2.putText(image,label,(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

                if labels[classIds[i]] == "person":
                    x,y,w,h = boxes[i]
                    if i%2==0:
                        color=(0,255,0)

                    else:
                        color=(255,0,0)
                    cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
                    label=labels[classIds[i]]
                    cv2.putText(image,label,(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

                




        cv2.imshow("basketball",image)
        cv2.waitKey(1)
    key=cv2.waitKey(1)
    if key== 32:
        print("Stopped")
        break
    if key == 112: # p key
        state="pause"
        
    if key == 108: #1 key
        state="play"                                

video.release()
cv2.destroyAllWindows()
