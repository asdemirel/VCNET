import sys
import json
import numpy as np
import cv2
from keras.models import load_model

import matplotlib.pyplot as plt
from PIL import Image

class tracking(object):
    def __init__(self,laneNum,threshold=0.45):
        self.laneStates = []
        self.threshold = threshold
        for i in range(laneNum):
            self.laneStates.append({"state":False})

    def check_passage(self,laneIdx,pred):
        laneInfo = self.laneStates[laneIdx]
        if pred >= self.threshold and laneInfo["state"]==False:
            laneInfo["state"] = True
        elif pred < self.threshold and laneInfo["state"]==True:
            laneInfo["state"] = False
            return 1
        return 0

class movingAverage(object):
    def __init__(self,laneNum,kernel_size=5):
        self.lane_Data = np.zeros((laneNum,kernel_size))

    def calc_average(self,new_data):
        self.lane_Data=np.roll(self.lane_Data,-1)
        self.lane_Data[:,-1] = new_data
        return np.average(self.lane_Data,axis=-1)

def draw_lines(frame,all_lines,pred,threshold):
    red_color = (0,0,255)
    green_color = (0,255,0)
    thickness = 2
    for i,line in enumerate(all_lines):
        for j,(x1,y1,x2,y2) in enumerate(line):
            if(pred[i][j]>threshold):
                frame = cv2.line(frame, (x1,y1), (x2,y2), green_color, thickness)
            else:
                frame = cv2.line(frame, (x1,y1), (x2,y2), red_color, thickness)

    return frame

def find_slop(p1:list,p2:list):
    m = (p2[1]-p1[1])/(p2[0]-p1[0])
    return m

def selectLines(points:list,nline):
    lines = []
    ylines = np.linspace(points[0][1],points[3][1],num=nline)
    m1 = find_slop(points[0],points[3])
    m2 = find_slop(points[1],points[2])
    for y1 in ylines:
        y1 = int(y1)
        x1 = ((y1-points[0][1])/m1)+points[0][0]
        x1 = int(x1)
        y2 = y1
        x2 = int(((y2-points[1][1])/m2)+points[1][0])
        lines.append((x1,y1,x2,y2))
    return lines


def load_config(jsonPath:str,nline=7):
    all_lines = []
    f = open(jsonPath,)
    json_data = json.load(f)
    for _id in range(len(json_data)-1):
        points = json_data[_id]["Points"]
        lines = selectLines(points,nline)
        all_lines.append(lines)
    return np.array(all_lines)
    

def read_video(video_path:str)->cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    return cap


def predict(frame,model,all_lines):
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    all_row_data = np.zeros((all_lines.shape[0]*all_lines.shape[1],128))
    idx = 0
    for line in all_lines:
        for x1,y1,x2,y2 in line:
            if x1<x2:
                nrow = frame[y1:y2+1,x1:x2]
            else:
                nrow = frame[y1:y2+1,x1:x2]
            nrow = cv2.resize(nrow,(128,1))
            all_row_data[idx:idx+1,:] = nrow
            idx += 1
    pred = model.predict(all_row_data.reshape(-1,128,1))[...,1]
    pred = pred.reshape((all_lines.shape[0],all_lines.shape[1]))
    avg = np.average(pred,axis=1)
    return avg,pred

if __name__ == "__main__":
    import time
    # videoPath = r"E:\ELOHARP\WORK\VCNET-Eren\videos\M-30.avi"
    # jsonPath = r"E:\ELOHARP\WORK\VCNET-Eren\M-30.json"
    # modelPath = r"E:\ELOHARP\WORK\VCNET-Eren\model\vehicle_counter.h5 "

    videoPath = r"E:\ELOHARP\WORK\VCNET-Eren\videos\M-30.avi"
    jsonPath = r"E:\ELOHARP\WORK\VCNET-Eren\M-30.json"
    modelPath = r"E:\ELOHARP\WORK\VCNET-Eren\model\vehicle_counter.h5 "

    nline = 7
    all_lines = load_config(jsonPath,nline=nline)

    v = read_video(videoPath)
    model = load_model(modelPath)

    total_frame = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    lane_size = all_lines.shape[0]
    avg_list = np.zeros((lane_size,total_frame))
    cycle_counter = 0
    counter = 0
    track = tracking(lane_size,threshold=0.35)
    moAv = movingAverage(lane_size,5)
    average_pred_time = 0

    while(cycle_counter<total_frame):
        ret,frame = v.read()
        if not ret:
            break
        s = time.time()
        avg,pred =  predict(frame,model,all_lines)
        avg=moAv.calc_average(avg)
        for idx in range(lane_size):
            isPassed = track.check_passage(idx,avg[idx])
            if isPassed == 1:
                counter=counter+1
        average_pred_time += (time.time()-s)*1000
        avg_list[:,cycle_counter] = avg
        draw_lines(frame,all_lines,pred,track.threshold)
        cv2.putText(frame, str(counter),(10, 50),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
        cycle_counter += 1
        frame = cv2.resize(frame,(1024,720))
        cv2.imshow("VCNET",frame)
        key = cv2.waitKey(1) & 0xFF   
        if key == 27:
            break
    print(f"Average pred time : {average_pred_time/(cycle_counter)}")
    fig, ax = plt.subplots(lane_size)
    fig.suptitle('Avg Values')
    for i in range(lane_size):
        ax[i].set_ylim([0,1])
        ax[i].plot(avg_list[i,:])
    cv2.destroyAllWindows()
    plt.show()
