# load libraries
from scipy.spatial import distance as dist
import numpy as np
import cv2
import imutils
import io
from PIL import Image
from flask import Flask,jsonify,request,send_file
from dotenv import load_dotenv
import json
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Define functions

weightsPath = 'yolov3.weights'

configPath = 'yolov3.cfg'

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

with open('coco.names','r') as file:
    LABELS = file.read().strip().split("\n")
        
Min_Confidence= 0.3
NMS_Threshold= 0.3

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def detect_people(frame, net, ln, personIdx=0):
    (Height, Width) = frame.shape[:2]
    results = []
    #Constructing a blob from the input frame and performing a forward pass of the YOLO object detector
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    centroids = []
    confidences = []
    
    #Looping over each of the layer outputs
    for output in layerOutputs:
    #Looping over each of the detections
        for detection in output:
            #Extracting the class ID and confidence of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            #Filtering detections by:
            #1 Ensuring that the object detected was a person
            #2 Minimum confidence is met
            if classID == personIdx and confidence > Min_Confidence:
                #Scaling the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([Width, Height, Width, Height])
                (centerX, centerY, width, height) = box.astype("int")
                #Using the center (x, y)-coordinates to derive the top and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                #Updating the list of bounding box coordinates, centroids, and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
                
    #Applying non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, Min_Confidence, NMS_Threshold)
    #ensuring at least one detection exists
    if len(idxs) > 0:
        #Looping over the indexes we are keeping
        for i in idxs.flatten():
            #Extracting the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            #Updating our results list to consist of the person prediction probability, bounding box coordinates and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    return results

def img_transform(Image_name):
    list_of_red = []
    list_of_green = []
    # frame = cv2.imread(Image_name)
    frame = Image_name
    #Resizing the frame and then detecting people in it
    frame = imutils.resize(frame, width=700)
    

    
    results = detect_people(frame, net, ln,
        personIdx=LABELS.index("person"))

    violate = set()

    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                #Checking if the distance between< number of pixels (60)
                if D[i, j] < 60:
                    violate.add(i)
                    violate.add(j)

    #Looping over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        #Extract the bounding box and centroid coordinates, colour set to blue if okay
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (255, 0, 0)

        #Red if in violation
        if i in violate:
            color = (0, 0, 255)
            #Bounding box and centroid marking
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            list_of_red.append([startX,startY,endX, endY])
        else:
            #Bounding box and centroid marking
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            list_of_green.append([startX,startY,endX, endY])

    dict_corr={"Red_corrdinates":list_of_red,"Green_corrdinates":list_of_green}
    return dict_corr


def get_4_windows(corr):
    left_bottem,left_top,right_bottem,right_top=[],[],[],[]

    for i in corr['Red_corrdinates']:
        x1 = i[0]
        x2 = i[1]
        x3 = i[2]
        x4 = i[3]
        # set-up filter to accuratelly recognize which conrrdinates belongs to which window
        # box 1
        if x1 > 350 and x2 < 262 and x3 > 350 and x4 < 262:
            right_top.append([x1,x2,x3,x4])
        elif x1 > 350 and x2 < 262:
            right_top.append([x1,x2,x3,x4])
        elif x3 > 350 and x4 < 262:
            right_top.append([x1,x2,x3,x4])

        # box 2
        if x1 > 350 and x2 > 262 and x3> 350 and x4 > 262:
            right_bottem.append([x1,x2,x3,x4])
        elif x1 > 350 and x2 > 262:
            right_bottem.append([x1,x2,x3,x4])
        elif x3> 350 and x4 > 262:
            right_bottem.append([x1,x2,x3,x4])

        # box 3
        if x1 < 350 and x2 < 262 and x3 < 350 and x4 < 262:
            left_top.append([x1,x2,x3,x4])
        elif x1 < 350 and x2 < 262:
            left_top.append([x1,x2,x3,x4])
        elif x3 < 350 and x4 < 262:
            left_top.append([x1,x2,x3,x4])

        # box 4
        if x1 < 350 and x2 > 262 and x3 < 350 and x4 > 262:
            left_bottem.append([x1,x2,x3,x4])
        elif x1 < 350 and x2 > 262:
            left_bottem.append([x1,x2,x3,x4])
        elif x3 < 350 and x4 > 262:
            left_bottem.append([x1,x2,x3,x4])
            
    return left_bottem,left_top,right_bottem,right_top

# remove overlapping corrdinates
def filter_4_windows(left_top,left_bottem,right_top,right_bottem):
    # left top and left  bottem
    LEFT_TB = [i for i in left_top if i in left_bottem] # get duplicate corrdinates
    for i in LEFT_TB:
        # get distance of each corrdinate with window border corrrdinate
        lt = abs(i[1]-262) 
        lb = abs(i[3]-262)
        try:
            if lt > lb:
                left_bottem.remove(i)
            else:
                left_top.remove(i)
        except:
            pass
        
    RIGHT_TB =  [i for i in right_top if i in right_bottem] # get duplicate corrdinates
    for i in RIGHT_TB:
        # get distance of each corrdinate with window border corrrdinate
        rt = abs(i[1]-262)
        rb = abs(i[3]-262)
        try:
            if rt > rb:
                right_bottem.remove(i)
            else:
                right_top.remove(i)
        except:
            pass
    
    LEFT_TOP_RIGHT_TOP = [i for i in right_top if i in left_top] # get duplicate corrdinates
    for i in LEFT_TOP_RIGHT_TOP:
        # get distance of each corrdinate with window border corrrdinate
        LF_T = abs(i[0]-350)
        RF_T = abs(i[2]-350)
        try:
            if LF_T > RF_T:
                right_top.remove(i)
            else:
                left_top.remove(i)
        except:
            pass
    
    LEFT_BOTTEM_RIGHT_BOTTEM = [i for i in right_bottem if i in left_bottem] # get duplicate corrdinates
    for i in LEFT_BOTTEM_RIGHT_BOTTEM:
        # get distance of each corrdinate with window border corrrdinate
        LF_B = abs(i[0]-350)
        RF_B = abs(i[2]-350)
        try:
            if LF_B > RF_B:
                right_bottem.remove(i)
            else:
                left_bottem.remove(i)
        except:
            pass
    
    return left_top,left_bottem,right_top,right_bottem   
        
def filter_9_windows(win1,win2,win3,win4,win5,win6,win7,win8,win9):
    window_1_2 = [i for i in win1 if i in win2] # get duplicate corrdinates
    for i in window_1_2:
        w1 = abs(i[1]-233) 
        w2 = abs(i[3]-233)
        try:
            if w1 > w2:
                win2.remove(i)
            else:
                win1.remove(i)
        except:
            pass

    window_2_3 = [i for i in win2 if i in win3] # get duplicate corrdinates
    for i in window_2_3:
        w2 = abs(i[1]-466) 
        w3 = abs(i[3]-466)
        try:
            if w2 > w3:
                win3.remove(i)
            else:
                win2.remove(i)
        except:
            pass

    window_4_5 = [i for i in win4 if i in win5] # get duplicate corrdinates
    for i in window_4_5:
        w4 = abs(i[1]-233) 
        w5 = abs(i[3]-233)
        try:
            if w4 > w5:
                win5.remove(i)
            else:
                win4.remove(i)
        except:
            pass

    window_5_6 = [i for i in win5 if i in win6] # get duplicate corrdinates
    for i in window_5_6:
        w5 = abs(i[1]-466) 
        w6 = abs(i[3]-466)
        try:
            if w5 > w6:
                win6.remove(i)
            else:
                win5.remove(i)
        except:
            pass

    window_7_8 = [i for i in win7 if i in win8] # get duplicate corrdinates
    for i in window_7_8:
        w7 = abs(i[1]-233) 
        w8 = abs(i[3]-233)
        try:
            if w7 > w8:
                win8.remove(i)
            else:
                win7.remove(i)
        except:
            pass

    window_8_9 = [i for i in win8 if i in win9] # get duplicate corrdinates
    for i in window_8_9:
        w8 = abs(i[1]-466) 
        w9 = abs(i[3]-466)
        try:
            if w8 > w9:
                win9.remove(i)
            else:
                win8.remove(i)
        except:
            pass

    window_1_4 = [i for i in win1 if i in win4] # get duplicate corrdinates
    for i in window_1_4:
        w1 = abs(i[1]-175) 
        w4 = abs(i[3]-175)
        try:
            if w1 > w4:
                win4.remove(i)
            else:
                win1.remove(i)
        except:
            pass

    window_4_7 = [i for i in win4 if i in win7] # get duplicate corrdinates
    for i in window_4_7:
        w4 = abs(i[1]-350) 
        w7 = abs(i[3]-350)
        try:
            if w4 > w7:
                win7.remove(i)
            else:
                win4.remove(i)
        except:
            pass

    window_2_5 = [i for i in win2 if i in win5] # get duplicate corrdinates
    for i in window_2_5:
        w2 = abs(i[1]-175) 
        w5 = abs(i[3]-175)
        try:
            if w2 > w5:
                win5.remove(i)
            else:
                win2.remove(i)
        except:
            pass

    window_5_8 = [i for i in win5 if i in win8] # get duplicate corrdinates
    for i in window_5_8:
        w5 = abs(i[1]-350) 
        w8 = abs(i[3]-350)
        try:
            if w5 > w8:
                win8.remove(i)
            else:
                win5.remove(i)
        except:
            pass

    window_3_6 = [i for i in win3 if i in win6] # get duplicate corrdinates
    for i in window_3_6:
        w3 = abs(i[1]-175) 
        w6 = abs(i[3]-175)
        try:
            if w3 > w6:
                win6.remove(i)
            else:
                win3.remove(i)
        except:
            pass

    window_6_9 = [i for i in win6 if i in win9] # get duplicate corrdinates
    for i in window_6_9:
        w6 = abs(i[1]-350) 
        w9 = abs(i[3]-350)
        try:
            if w6 > w9:
                win9.remove(i)
            else:
                win6.remove(i)
        except:
            pass
    return win1,win2,win3,win4,win5,win6,win7,win8,win9

def get_9_windows(corr):
    win1,win2,win3,win4,win5,win6,win7,win8,win9=[],[],[],[],[],[],[],[],[]
    for i in corr['Red_corrdinates']:
        x1 = i[0]
        x2 = i[1]
        x3 = i[2]
        x4 = i[3]
        ###################################
        ############# box 1 ###############
        ###################################
        if x1 > 0 and x2 > 0 and x3 < 233 and x4 < 175:
            win1.append([x1,x2,x3,x4])
        elif x1 < 233 and x2 < 175:
            win1.append([x1,x2,x3,x4])
        elif x3 < 233 and x4 < 175:
            win1.append([x1,x2,x3,x4])

        ###################################
        ############# box 2 ###############
        ###################################
        if x1 > 233 and x2 > 0 and x3 < 466 and x4 <175:
            win2.append([x1,x2,x3,x4])
        elif x1 > 233 and x1 < 466 and x2 >0 and x2 < 175:
            win2.append([x1,x2,x3,x4])
        elif x3 > 233 and x3 < 466 and x4 >0 and x4 < 175:
            win2.append([x1,x2,x3,x4])

        ###################################
        ############# box 3 ###############
        ###################################
        if x1 > 466 and x2 > 0 and x3 < 699 and x4 < 175:
            win3.append([x1,x2,x3,x4])
        elif x1 > 466 and x1 < 699 and x2 > 0 and x2 < 175:
             win3.append([x1,x2,x3,x4])
        elif x3 > 466 and x3 < 699 and x4 > 0 and x4 < 175:
             win3.append([x1,x2,x3,x4])

        ###################################
        ############# box 4 ###############
        ###################################
        if x1 > 0 and x2 > 175 and x3 <233 and x4 < 350:
            win4.append([x1,x2,x3,x4])
        elif x1 >0 and x1 < 233 and x2 > 175 and x2 < 250:
            win4.append([x1,x2,x3,x4])
        elif x3 >0 and x3 < 233 and x4 > 175 and x4 < 250:
            win4.append([x1,x2,x3,x4])

        ###################################
        ############# box 5 ###############
        ###################################
        if x1 > 233 and x2 > 175 and x3 < 466 and x4 < 350:
            win5.append([x1,x2,x3,x4])
        elif x1 > 233 and x1 < 466 and x2 > 175 and x2 < 350:
            win5.append([x1,x2,x3,x4])
        elif x3 > 233 and x3 < 466 and x4 > 175 and x4 < 350:
            win5.append([x1,x2,x3,x4])

        ###################################
        ############# box 6 ###############
        ###################################
        if x1 > 466 and x2 > 175 and x3 <699 and x4 < 350:
            win6.append([x1,x2,x3,x4])
        elif x1 > 466 and x1 < 699 and x2 > 175 and x2 < 350:
            win6.append([x1,x2,x3,x4])
        elif x3 > 466 and x3 < 699 and x4 > 175 and x4 < 350:
            win6.append([x1,x2,x3,x4])

        ###################################
        ############# box 7 ###############
        ###################################
        if x1 > 0 and x2 > 350 and x3 < 233 and x4 < 525:
            win7.append([x1,x2,x3,x4])
        elif x1 > 0 and x1 < 233 and x2 > 350 and x2 < 525:
            win7.append([x1,x2,x3,x4])
        elif x1 > 0 and x1 < 233 and x2 > 350 and x2 < 525:
            win7.append([x1,x2,x3,x4])

        ###################################
        ############# box8 ###############
        ###################################
        if x1 > 233 and x2 > 350 and x3 < 466 and x4 < 525:
            win8.append([x1,x2,x3,x4])
        elif x1 > 233 and x1 < 466 and x2 > 350 and x2 < 525:
            win8.append([x1,x2,x3,x4])
        elif x3 > 233 and x3 < 466 and x4 > 350 and x4 < 525:
            win8.append([x1,x2,x3,x4])

        ###################################
        ############# box9 ###############
        ###################################
        if x1 > 466 and x2 > 350 and x3 < 699 and x4 < 525:
            win9.append([x1,x2,x3,x4])
        elif x1 > 466 and x1 < 699 and x2 > 350 and x2 < 525:
            win9.append([x1,x2,x3,x4])
        elif x3 > 466 and x3 < 699 and x4 > 350 and x4 < 525:
            win9.append([x1,x2,x3,x4])
    return win1,win2,win3,win4,win5,win6,win7,win8,win9

def visulize_heatmap_9_win(win1,win2,win3,win4,win5,win6,win7,win8,win9):
    try:        
        avg_count_1 = round(mean([i for i in range(1,len(win1)+1)]))
    except:
        avg_count_1 = 0
    try:
        avg_count_2 = round(mean([i for i in range(1,len(win2)+1)]))
    except:
        avg_count_2 = 0
    try:
        avg_count_3 = round(mean([i for i in range(1,len(win3)+1)]))
    except:
        avg_count_3 = 0
    try:
        avg_count_4 = round(mean([i for i in range(1,len(win4)+1)]))
    except:
        avg_count_4 = 0
    try:
        avg_count_5 = round(mean([i for i in range(1,len(win5)+1)]))
    except:
        avg_count_5 = 0
    try:
        avg_count_6 = round(mean([i for i in range(1,len(win6)+1)]))
    except:
        avg_count_6 = 0
    try:
        avg_count_7 = round(mean([i for i in range(1,len(win7)+1)]))
    except:
        avg_count_7 = 0
    try:
        avg_count_8 = round(mean([i for i in range(1,len(win8)+1)]))
    except:
        avg_count_8 = 0
    try:
        avg_count_9 = round(mean([i for i in range(1,len(win9)+1)]))
    except:
        avg_count_9 = 0

    df = pd.DataFrame({"Symbol":["Average count of \n Window 1:","Average count of \n Window 2:","Average count of \n Window 3:",
                                 "Average count of \n Window 4:","Average count of \n Window 5:","Average count of \n Window 6:",
                                 "Average count of \n Window 7:","Average count of \n Window 8:","Average count of \n Window 9:"],
                       "Change":[avg_count_1,avg_count_2,avg_count_3,avg_count_4,avg_count_5,avg_count_6,
                                avg_count_7,avg_count_8,avg_count_9],"Yrows":[1,1,1,2,2,2,3,3,3],
                       "Xcols":[1,2,3,1,2,3,1,2,3]})


    symbol = ((np.asarray(df['Symbol'])).reshape(3,3))
    perchange = ((np.asarray(df['Change'])).reshape(3,3))

    result = df.pivot(index='Yrows', columns='Xcols', values='Change')
    labels = (np.asarray(["{0} \n {1:.2f}".format(symb, value) for symb, value in zip(symbol.flatten(), perchange.flatten())])).reshape(3, 3)

    fig, ax = plt.subplots(figsize=(13, 7))

    title = "Heatmap Analysis"
    plt.title(title, fontsize=18)
    ttl = ax.title
    ttl.set_position([0.5, 1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    sns.heatmap(result, annot=labels, fmt="", cmap='bwr', linewidths=0.40, ax=ax)
    
    plt.savefig('svm_conf.png', dpi=500)
    # plt.show()
    
def visulize_heatmap_4_win(left_top,left_bottem,right_top,right_bottem):
    try:
        avg_count_RT = round(mean([i for i in range(1,len(right_top)+1)]))
    except:
        avg_count_RT = 0
    try:
        avg_count_RB = round(mean([i for i in range(1,len(right_bottem)+1)]))
    except:
        avg_count_RB = 0
    try:
        avg_count_LT = round(mean([i for i in range(1,len(left_top)+1)]))
    except:
        avg_count_LT = 0
    try:
        avg_count_LB = round(mean([i for i in range(1,len(left_bottem)+1)]))
    except:
        avg_count_LB= 0
    df = pd.DataFrame({"Symbol":["Average count of \n Top Left Window","Average count of \n Top Right Window","Average count of \n Bottem Left Window","Average count of \n Bottem Right Window"],"Change":[avg_count_LT,avg_count_RT,avg_count_LB,avg_count_RB],"Yrows":[1,1,2,2],"Xcols":[1,2,1,2]})

    symbol = ((np.asarray(df['Symbol'])).reshape(2,2))
    perchange = ((np.asarray(df['Change'])).reshape(2,2))

    result = df.pivot(index="Yrows",columns="Xcols",values="Change")

    # create an array to annotate heatmap
    labels = (np.asarray(["{0} \n {1:2f}".format(symb,value)
                         for symb, value in zip(symbol.flatten(),
                                               perchange.flatten())])
             ).reshape(2,2)
    for i in labels:
        i[0] = i[0][:-3]
        i[1] = i[1][:-3]

    # heatap#############################
    #####################################
    fig, ax = plt.subplots(figsize=(10,5))
    title = "Heatmap Analysis"
    plt.title(title,fontsize=17)
    ttl = ax.title
    ttl.set_position([0.5,1.05])

    ax.set_xticks([])
    ax.set_yticks([])

    ax.axis('off')

    sns.heatmap(result,annot=labels,fmt="",cmap="bwr",linewidths=2,ax=ax)

    plt.savefig('svm_conf.png', dpi=500)
    # plt.show()
    
def heatmap_visulization(windows,corr):
    if windows == "4":
        # 4 windows
        left_bottem,left_top,right_bottem,right_top = get_4_windows(corr)
        left_top,left_bottem,right_top,right_bottem = filter_4_windows(left_top,left_bottem,right_top,right_bottem)
        visulize_heatmap_4_win(left_top,left_bottem,right_top,right_bottem)
        graph_image = cv2.imread("svm_conf.png")
        return True,graph_image
    elif windows == "9":
        # 9 windows
        win1,win2,win3,win4,win5,win6,win7,win8,win9 = get_9_windows(corr)
        win1,win2,win3,win4,win5,win6,win7,win8,win9 = filter_9_windows(win1,win2,win3,win4,win5,win6,win7,win8,win9)
        visulize_heatmap_9_win(win1,win2,win3,win4,win5,win6,win7,win8,win9)
        graph_image = cv2.imread("svm_conf.png")
        return True,graph_image
    else:
        return False
    
def main_header(image_path,grid_tiles):
    # get corrdinates and 4 slice of image
    corr = img_transform(image_path) # get four parts
    status,cv2_image = heatmap_visulization(grid_tiles,corr)
    return status,cv2_image


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
# set env for secret key
load_dotenv()

secret_id = os.getenv('AI_SERVICE_SECRET_KEY')

def check_for_secret_id(request_data):    
    try:
        if 'secret_id' not in request_data.keys():
            return False, "Secret Key Not Found."
        
        else:
            if request_data['secret_id'] == secret_id:
                return True, "Secret Key Matched"
            else:
                return False, "Secret Key Does Not Match. Incorrect Key."
    except Exception as e:
        message = "Error while checking secret id: " + str(e)
        return False,message

@app.route('/heatmap_visulization',methods=['POST'])  #main function
def main():
    key = request.form['secret_id']
    grid_tiles = request.form['grid']
    request_data = {'secret_id' : key}
    secret_id_status,secret_id_message = check_for_secret_id(request_data)
    print ("Secret ID Check: ", secret_id_status,secret_id_message)
    if not secret_id_status:
        return jsonify({'message':"Secret Key Does Not Match. Incorrect Key.",
                        'success':False}) 
    else:
        img_params =request.files['image'].read()
        npimg = np.fromstring(img_params, np.uint8)
        #load image
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        st, img = main_header(image,grid_tiles)

        I = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(I.astype('uint8'))
        file_object = io.BytesIO()
        img.save(file_object, 'PNG')
        file_object.seek(0)

        output = send_file(file_object, mimetype='image/PNG') 

        # remove file
        file_mg = glob.glob('svm_conf.png');
        try:
            for f in file_mg:
                os.remove(f)
        except:
            pass

    return output
    
if __name__ == '__main__':
    app.run()                       
                