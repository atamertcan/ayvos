import cv2 as cv
import mediapipe as mp
import numpy as np


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
            h, w, c = frame.shape
            knee = [int(landmarks[self.mpPose.PoseLandmark.LEFT_KNEE.value].x * w),
                    int(landmarks[self.mpPose.PoseLandmark.LEFT_KNEE.value].y * h)]
            hip = [int(landmarks[self.mpPose.PoseLandmark.LEFT_HIP.value].x * w),
                   int(landmarks[self.mpPose.PoseLandmark.LEFT_HIP.value].y * h)]
            shoulder = [int(landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                        int(landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value].y * h)]
            lmList = [knee, hip, shoulder]

        return lmList

    def calculateAngle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180 / np.pi)
        if angle > 180:
            angle = 360 - angle

        return angle


def main():
    capture = cv.VideoCapture("../Videos/sitandstand.mp4")
    detector = poseDetector()
    while True:
        isTrue, frame = capture.read()
        frame = detector.findPose(frame)
        lmList = detector.findPoints(frame)
        angle = detector.calculateAngle(lmList[0], lmList[1], lmList[2])
        if len(lmList) == 3:
            cv.putText(frame,f"{int(angle)} deg", (50,125),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if angle > 145:
            cv.putText(frame, f"Standing", (50, 75),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        else:
            cv.putText(frame, f"Sitting", (50, 75),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        cv.imshow("Img", frame)
        cv.waitKey(1)

if __name__ == "__main__":
    main()