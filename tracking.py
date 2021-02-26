'''
Only tracking, no reidentification.
Tracking is done using last k=15 frames.
'''
import numpy as np
import tensorflow as tf
import cv2
import time
import os

from importlib import import_module

iou_threshold = 0.4
#maximum number of previous frames to check iou with
k = 25

def iou(box1, box2):
    xa = max( box1[1] , box2[1] )
    ya = max( box1[0] , box2[0] )
    xb = min( box1[3] , box2[3] )
    yb = min( box1[2] , box2[2] )
    
    interArea = max(0, xb - xa ) * max(0, yb - ya )

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1] )
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1] )
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = float(interArea) / float(box1Area + box2Area - interArea)

    # return the intersection over union value
    return iou
 
def track():
    
    #this will store the bounding boxes detected in the previous frame.
    boxes_prev = []
    framenum = 1
    #iterate over frames
    while True: 
        r, img = cap.read()
        img = cv2.resize(img, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)
        boxes_cur = []
        for l in range(len(boxes_prev)):
            if( len(boxes_prev[l]) < k ):
                boxes_cur.append(  [-1] + boxes_prev[l]  )
            else:
                boxes_cur.append(  [-1] + boxes_prev[l][0:k-1]  )
                
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                
                #draw the bounding box on the image
                # cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                
                cropped_img = img[ box[0]:box[2] , box[1]:box[3] ]
                
                maxthreshold = -1
                maxindex = 101     #the index in boxes_prev indicating the matching person from the previous k frames. 
                
                for j in range( len(boxes_prev) ):
                    #Every boxes_prev[j] denotes a person. It is a list of the last k positions of the person j.
                    
                    if( boxes_prev[j] == -1 ):   #This previous person has already been alloted to another person in the current frame 
                        continue
                    
                    for kk in range( len(boxes_prev[j]) ):
                        if(boxes_prev[j][kk] == -1):      #person was not detected in frame kk
                            continue
                        r = iou( boxes_prev[j][kk] ,box)
                        if(  r > maxthreshold  and  r > iou_threshold):
                            maxthreshold = r
                            maxindex = j            
                        
                    
                #maxthreshold != -1 at this point means this person is the same as prevbox in the last frame. 
                if( maxthreshold != -1 ):
                    boxes_cur[ maxindex ][0] = box
                    boxes_prev[ maxindex ] = -1
                    
                    #also add this image of the person to his previous images
                    cv2.putText(img, str(maxindex), (box[1],box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA )
                else:
                    boxes_cur.append( [box] )
   
        print('#People:   ' + str(len(boxes_cur)))
        print(' ')        
          
        framenum += 1  
        boxes_prev =  boxes_cur

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

