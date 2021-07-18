import cv2
import mediapipe as mp
import time

'''
Hand Landmarks:
  WRIST = 0
  THUMB_CMC = 1
  THUMB_MCP = 2
  THUMB_IP = 3
  THUMB_TIP = 4
  INDEX_FINGER_MCP = 5
  INDEX_FINGER_PIP = 6
  INDEX_FINGER_DIP = 7
  INDEX_FINGER_TIP = 8
  MIDDLE_FINGER_MCP = 9
  MIDDLE_FINGER_PIP = 10
  MIDDLE_FINGER_DIP = 11
  MIDDLE_FINGER_TIP = 12
  RING_FINGER_MCP = 13
  RING_FINGER_PIP = 14
  RING_FINGER_DIP = 15
  RING_FINGER_TIP = 16
  PINKY_MCP = 17
  PINKY_PIP = 18
  PINKY_DIP = 19
  PINKY_TIP = 20
'''

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionConf=0.5,trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self,img, handnum=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            handLms = self.results.multi_hand_landmarks[handnum]
            
            for idx, lm in enumerate(handLms.landmark):
                    #print(idx,lm)
                    
                    # rudimentary landmark tracker
                    h,w,c = img.shape
                    cx,cy = int(lm.x*w),int(lm.y*h)

                    lmList.append([idx,cx,cy])

                    if draw:
                        cv2.circle(img,(cx,cy),6,(255,0,255),cv2.FILLED)
        return lmList


def main():
    pTime = 0
    cTime = 0
    
    cap = cv2.VideoCapture(0)

    detector = handDetector()
    
    while True:
        success, img = cap.read()
        
        img = detector.find_hands(img)

        mylist = detector.find_position(img)
        if len(mylist) != 0:
            print(mylist[3])
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime


        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),3)
        cv2.imshow("Image",img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()