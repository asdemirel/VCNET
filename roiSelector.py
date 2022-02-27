import numpy as np
import argparse
import json
import sys
import cv2

def printInfos():
    print('[INFO] This is a Multiple ROI selector script for multiple images')
    print('[INFO] Click the left button: select the point, right click: delete the last selected point')
    print('[INFO] Press ‘W’ to  view next frame')
    print('[INFO] Press ‘S’ to  save all ROIs')
    print('[INFO] Press ‘D’ to delete last selected ROI')
    print('[INFO] Press ESC to quit')
    
def plotSegmentedRois(imTemp, imMask,rois, colors):
    for qq, rs in enumerate(rois):
        roiPoints = rs['Points']
        roiPoints = np.array(roiPoints, np.int32)
        roiPoints = roiPoints.reshape((-1, 1, 2))
        imMask = cv2.fillPoly(imMask, [roiPoints], colors[qq%len(colors)])
    imTemp = cv2.addWeighted(imTemp, 0.5, imMask, 0.5, 0)
    return imTemp, imMask

def checkRoiClosed(pts,distTh=5):
    points = np.array(pts, np.int32)
    dist = np.linalg.norm(points[0] - points[-1])
    if dist < distTh:
        return True
    return False

def drawRoi(event, x, y,  flags, param):
    global pts, id , click_counter
    colors = [(255,0,0),(0,255,0), (0,0,255), (204,78,1), (79,0,128), (255,255,85), (79,186,218), (229,152,230), (88,0,144)]
    imTemp = im.copy()
    imMask = im.copy()


    if event == cv2.EVENT_LBUTTONDOWN:  
        if(click_counter)==5:
            pts.append((x, y))  
            click_counter = 0
        elif(click_counter%2==1):
            pts.append((x, pts[-1][-1])) 
        else:
            pts.append((x, y))  
        click_counter += 1        

    if event == cv2.EVENT_RBUTTONDOWN:  
        if len(pts)>0:
            pts.pop()  
            click_counter -= 1
    
    imTemp, imMask = plotSegmentedRois(imTemp, imMask,rois, colors)

    if len(pts) > 0:
        cv2.circle(imTemp, pts[-1], 1, (0, 0, 255), -1)


    if len(pts) > 1:

        if checkRoiClosed(pts,distTh=5):
            rois.append({'Id': id, 'Points': pts})
            id += 1
            pts = []
    
        for i in range(len(pts) - 1):
            cv2.circle(imTemp, pts[i], 1, (0, 0, 255), -1)
            cv2.line(imTemp, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=1)
    
    cv2.imshow('image', imTemp)

def roiSelector(image):
    global im, key, pts, rois, id, json_name,click_counter

    id = 1
    pts = []
    rois = []
    click_counter = 0

    h,w = image.shape[0],image.shape[1]
    im = image
    cv2.namedWindow('image',cv2.WINDOW_FREERATIO)
    cv2.imshow('image', im)
    cv2.setMouseCallback('image', drawRoi)    
    
    while True:

        key = cv2.waitKey(1) & 0xFF   

        if key == 27:
            return -1
            break

        if chr(key).lower() == 'd':
            if len(rois) < 1: continue
            rois.pop()
            pts = []
            id -= 1
            click_counter = 0
        if chr(key).lower() == 'w':
            break
            
        if chr(key).lower() == 's':
            rois.append({'Id': id, 'Points': pts})
            with open(json_name,'w') as jsonRoi:
                json.dump(rois, jsonRoi)
            return -1


def read_video(video_path:str)->cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    return cap

if __name__ == "__main__":
    printInfos()
  
    video_path = r"E:\ELOHARP\WORK\VCNET-Eren\videos\highway.avi"
    video_frames = read_video(video_path)
    json_name = video_path.split("\\")[-1].split(".")[0]+".json"
    print(json_name)
    while(video_frames.isOpened()):# Loop all images and run roi selector
        ret,image = video_frames.read(0)
        exit_val = roiSelector(image)
        if(exit_val == -1):
            break
    cv2.destroyAllWindows()
    