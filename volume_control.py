import cv2
import time
import numpy as np
import handtracker_module as htm
import math
import alsaaudio as aud

############# Pycaw #####################
#from ctypes import cast, POINTER
#from comtypes import CLSCTX_ALL
#from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#########
wCam ,hCam = 640, 480
#########

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

pTime = 0

detector = htm.handDetector(detectionConf=0.7)

m = aud.Mixer('Master')
print(m.getenum())

volume = aud.Mixer('Master')
volRange = volume.getrange()

print(volRange)
volume.setvolume(50)

minVol = volRange[0]
maxVol = volRange[1]

main_vol = 0
main_vol_bar = 280
while True:
    success, img = cap.read()
    img = detector.find_hands(img)

    lmlist = detector.find_position(img,draw=False)
    if len(lmlist) != 0:
        #index tip, thumb tip values: 
        #print(lmlist[4],lmlist[8])

        x1,y1 = lmlist[4][1],lmlist[4][2]
        x2,y2 = lmlist[8][1],lmlist[8][2]

        cx, cy = ((x1+x2)//2),((y1+y2)//2)

        cv2.circle(img,(x1,y1), 10, (255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2), 10, (255,0,255),cv2.FILLED)
        cv2.circle(img,(cx,cy), 5, (255,0,255),cv2.FILLED)
        cv2.line(img, (x1,y1),(x2,y2),(255,0,255),2)

        length = math.hypot(x2-x1,y2-y1)
        #print(length)
        
        # Hand range : 30 - 280
        # vol range: 0 - 65536
        # actual setvolume argument: 0 - 100
        
        vol = np.interp(length,[30,280],[minVol,maxVol])
        main_vol = np.interp(vol,[minVol,maxVol], [0,100])
        
        main_vol_bar = np.interp(main_vol,[0,100], [300,150])
        volPer = np.interp(length,[30,280],[0,100])
        
        print(int(vol),int(main_vol))
        
        volume.setvolume(int(main_vol))

        if length<50:
            cv2.circle(img,(cx,cy), 5, (255,0,0),cv2.FILLED)


        cv2.rectangle(img, (50,150),(85,300),(0,255,255),3)
        cv2.rectangle(img, (50,int(main_vol_bar)),(85,300),(0,255,255), cv2.FILLED)
        
        cv2.putText(img,f'Volume:{int(volPer)}%',(40,350),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),1)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    cv2.putText(img,f'FPS:{int(fps)}',(10,30),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),1)
    cv2.imshow("Image",img)

    cv2.waitKey(1)
