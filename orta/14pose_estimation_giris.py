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
                                           self.mpPose.POSE_CONNECTIONS,
                                           self.mpDraw.DrawingSpec(color=(255,0,0), thickness = 5, circle_radius = 5),
                                           self.mpDraw.DrawingSpec(color=(0,255,0), thickness = 5, circle_radius = 2)
                                           )
        return frame

    def findPoints(self, frame, draw=True):
        lmList=[]
        if self.results.pose_landmarks:
            landmarks = self.results.pose_landmarks.landmark
            #h, w, c = frame.shape
            #cx = int(landmarks[self.mpPose.PoseLandmark.RIGHT_WRIST].x * w)
            #cy = int(landmarks[self.mpPose.PoseLandmark.RIGHT_WRIST].y * h)
            #lmList.append([cx,cy])
            for id, lm in enumerate(landmarks):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w),int(lm.y * h)
                lmList.append([id,cx,cy])
        return lmList

def main():
    capture = cv.VideoCapture("../Videos/rope.mp4")
    detector = poseDetector()
    while True:
        isTrue, frame = capture.read()
        frame = detector.findPose(frame)
        lmList = detector.findPoints(frame)
        cv.circle(frame, (lmList[16][1],lmList[16][2]), 25,(0,0,255),cv.FILLED)

        cv.imshow("Img", frame)
        cv.waitKey(1)

if __name__ == "__main__":
    main()