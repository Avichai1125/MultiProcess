import cv2
import numpy as np
import multiprocessing as mp
from config import lk_params, colors, NUM_OF_FRAMES_TO_STACK, video_file_path

# mog background subtraction
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# read frames from video file
cap = cv2.VideoCapture(video_file_path)


def get_points_from_cnts(cnts):
    points = []
    bboxs = []
    for c in cnts:
        # print(cv2.contourArea(c))   # uncomment tis for testing
        # if the contour is too small or too big, ignore it
        if cv2.contourArea(c) < 500 or cv2.contourArea(c) > 50000:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        (x, y, w, h) = cv2.boundingRect(c)
        center_x, center_y = (x + x + w) / 2, (y + y + h) / 2
        points.append([[center_x, center_y]])  # insert the centroid
        bboxs.append((x, y, w, h))

    return np.array(points).astype("float32"), bboxs


#The function for the streamer process, outputs for each frame its corresponding old gray image.
def Streamer(frame):
    old_blur = cv2.GaussianBlur(frame, (21, 21), 0)
    fgmask = fgbg.apply(old_blur, learningRate=1)
    threshold_frame = cv2.dilate(fgmask, None, iterations=2)
    cnts = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    p0, _ = get_points_from_cnts(cnts[0])
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    return frame, old_gray

#The function for the detector process, outputs for each frame and its old gray vision
#the data for the bounding boxes and the optical flow
def Detector(frame,old_gray):
    old_blur = cv2.GaussianBlur(frame, (21, 21), 0)
    fgmask = fgbg.apply(old_blur, learningRate=1)
    threshold_frame = cv2.dilate(fgmask, None, iterations=2)
    cnts = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    p0, bboxs = get_points_from_cnts(cnts[0])

    if not p0.size:
        continue

    # calculate optical flow
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    return frame, good_new, good_old

#The function for the drawer process, outputs for each frame the full image with the annotations for the bounding boxes and the optimal flow
def Drawer(frame,good_new,good_old):
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 7, colors[i].tolist(), -1)

    # draw the bounding box
    for (x, y, w, h) in bboxs:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img = cv2.add(frame, mask)
    
    return img

#The main part application

#Exctracting the first frames of the movie
frames = []
for i in range(1, NUM_OF_FRAMES_TO_STACK):
    # Take first NUM_OF_FRAMES_TO_STACK frame and find corners in it
    ret, frame = cap.read()
    frames.append(frame)

# Create a mask image for drawing purposes
mask = np.zeros_like(frames[0])

#Initiating the streamer process
frame_tuples = mp.Pool.map(Streamer,frames)

#Inititating the detector process
frames_detect = mp.Pool.map(Detector,frame_tuples)

#Inititating the drawer process
final_images = mp.Pool.map(Drawer,frames_detect)

for img in final_images:
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
