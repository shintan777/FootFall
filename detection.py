import numpy as np
import tensorflow as tf
import cv2
import time

# from utils.misc import pre_reid_process
from pprint import pprint


MODEL_PATH = "detector/frozen_inference_graph.pb"

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


odapi = DetectorAPI(path_to_ckpt=MODEL_PATH)

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

def detect(img, threshold=0.63, iou_threshold=0.4, k = 25):
    """
    Parameters
    -----------
    threshold: float
        Confidence threshold for detection

    iou_threshold: float
        Connfidence threshold for tracking

    img: cv2 image object
        Single image frame

    k: int
        Maximum number of previous frames to check iou with
    """

    img = cv2.resize(img, (640, 480))
    boxes, scores, classes, num = odapi.processFrame(img)

    #changes from here
    boxes_prev = []
    framenum = 1
    boxes_cur = []
    ids = {}

    for l in range(len(boxes_prev)):
        if( len(boxes_prev[l]) < k ):
            boxes_cur.append(  [-1] + boxes_prev[l]  )
        else:
            boxes_cur.append(  [-1] + boxes_prev[l][0:k-1]  )

    for i in range(len(boxes)):
        # Class 1 represents human
        if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]

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
                person_id = maxindex
            else:
                boxes_cur.append( [box] )
                person_id = len(boxes_cur)

            #draw the bounding box on the image
            cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
            cv2.putText(img, str(person_id), (box[1],box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA )
            ids[person_id] = box

    boxes_prev =  boxes_cur
    framenum += 1

    return img, len(boxes_cur), ids
