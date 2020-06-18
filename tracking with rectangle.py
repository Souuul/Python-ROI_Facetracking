# Tracking with mouse roi.py
import cv2
import numpy as np


roi  = None
drag_start = None
mouse_status = 0
tracking_start  = False
def onMouse(event, x, y, flags, param=None):
     global roi
     global drag_start
     global mouse_status
     global tracking_start   
     if event == cv2.EVENT_LBUTTONDOWN:
          drag_start = (x, y)
          mouse_status = 1
          tracking_start = False
     elif event == cv2.EVENT_MOUSEMOVE:
          if flags == cv2.EVENT_FLAG_LBUTTON:
               xmin = min(x, drag_start[0])
               ymin = min(y, drag_start[1])
               xmax = max(x, drag_start[0])
               ymax = max(y, drag_start[1])
               roi = (xmin, ymin, xmax, ymax)
               mouse_status = 2 # dragging
     elif event == cv2.EVENT_LBUTTONUP:
          mouse_status = 3 # complete
          
cv2.namedWindow('tracking')
cv2.setMouseCallback('tracking', onMouse)

cap = cv2.VideoCapture(0)
if (not cap.isOpened()): 
     print('Error opening video')    
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
roi_mask   = np.zeros((height, width), dtype=np.uint8)
term_crit = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,10, 1)


# Kalman Filter setup
q = 1e-5   #  process noise covariance
r = 0.01 #  measurement noise covariance, r = 1
dt = 1
KF = cv2.KalmanFilter(4,2,0)
KF.transitionMatrix = np.array([[1,0,dt,0],         
                                [0,1,0,dt],
                                [0,0,1,0],
                                [0,0,0,1]], np.float32)  # A
KF.measurementMatrix = np.array([[1,0,0,0],
                                 [0,1,0,0]],np.float32)  # H
t = 0
while True:
     ret, frame = cap.read()
     if not ret: break
     t+=1
     print('t=',t)
     frame2 = frame.copy() # camShift
     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     mask = cv2.inRange(hsv,(0, 60, 32),(180,255,255))

     if mouse_status==2:
          x1, y1, x2, y2 = roi
          cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  
     if mouse_status==3:
          print('initialize....')
          mouse_status = 0
          x1, y1, x2, y2 = roi          
          mask_roi = mask[y1:y2, x1:x2]
          hsv_roi  =  hsv[y1:y2, x1:x2]
          
          hist_roi = cv2.calcHist([hsv_roi],[0],mask_roi,[16],[0,180])

          cv2.normalize(hist_roi,hist_roi,0,255,cv2.NORM_MINMAX)
          H1 = hist_roi.copy()
          cv2.normalize(H1,H1,0.0,1.0,cv2.NORM_MINMAX)          
          track_window = (x1, y1, x2-x1, y2-y1) # meanShift
          
# Kalman filter initialize    
          KF.processNoiseCov     = q* np.eye(4, dtype=np.float32) # Q        
          KF.measurementNoiseCov = r* np.eye(2, dtype=np.float32) # R
          KF.errorCovPost  = np.eye(4, dtype=np.float32)          # P0 = I

          x, y, w, h = track_window
          KF.statePost=np.array([[x],[y],[0.],[0.]],dtype=np.float32)
          tracking_start = True
               
     if tracking_start:
          predict  = KF.predict()

#meanShift tracking
          backP = cv2.calcBackProject([hsv],[0],hist_roi,[0,180],1)
          backP &= mask
         
          ret, track_window = cv2.meanShift(backP, track_window, term_crit)
          x,y,w,h = track_window
          cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),2)

# Kalman correct
          z = np.array([[x],[y]],dtype=np.float32) # measurement
          estimate = KF.correct(z)
          estimate = np.int0(estimate)
          
       
          x2, y2 = estimate[0][0], estimate[1][0]
          cv2.rectangle(frame, (x2,y2), (x2+w,y2+h), (255,0,0),2)

          
     cv2.imshow('tracking',frame)
     key = cv2.waitKey(25)
     if key == 27:
          break
if cap.isOpened():
     cap.release();
cv2.destroyAllWindows()
