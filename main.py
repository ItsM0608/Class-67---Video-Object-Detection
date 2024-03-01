import cv2
import numpy as np

# Path to model configuration and weights files
modelConfiguration='cfg/yolov3.cfg'
modelWeights='yolov3.weights'

labels=open("coco.names").read().strip().split('\n')
print(labels)

confidenceThreshold=0.5
# Load YOLO object detection network
yoloNetwork=cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Load image
image= cv2.imread('static/img1.jpg')
image= cv2.resize(image,(800,800))
# Get image dimensions
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
        x=boxes[i][0]
        y=boxes[i][1]
        w=boxes[i][2]
        h=boxes[i][3]
        cv2.rectangle(image, (x,y), (x+w,y+h),(0,0,255),2)
        label=labels[classIds[i]]

        text='{}:{:2f}'.format(label,confidences[i]*100)
        cv2.putText(image,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)


# Display image
cv2.imshow("image",image)
cv2.waitKey(0)
