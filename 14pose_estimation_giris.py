import cv2 as cv
import mediapipe as mp

class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionConf=0.5, trackConf=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionConf,
            min_tracking_confidence=self.trackConf
        )

    def findPose(self,frame, draw=True):

        frameRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.results = self.pose.process(frameRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return frame

    def findPoints(self, frame, draw=True):
        lmList=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w),int(lm.y * h)
                #lmName = self.mpPose.PoseLandmark(14).name id'nin ismini almak için kullanabilirsin
                lmList.append([id,cx,cy])
                cv.circle(frame, (cx,cy), 5,
                          (255,0,0), cv.FILLED)
        return lmList

def main():
    capture = cv.VideoCapture("Videos/rope.mp4")
    detector = poseDetector()
    while True:
        isTrue, frame = capture.read()
        frame = detector.findPose(frame)
        lmList = detector.findPoints(frame)
        cv.circle(frame, (lmList[14][1],lmList[14][2]), 25,(0,0,255),cv.FILLED)

        cv.imshow("Img", frame)
        cv.waitKey(1)

if __name__ == "__main__":
    main()