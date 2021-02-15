import os
import cv2
import sys

if __name__ == "__main__":

    for vid in os.listdir(sys.argv[1]):
        video = cv2.VideoCapture(os.path.join(sys.argv[1], vid))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        counter = 0
        errors = 0
        while video.isOpened():
            success, frame = video.read()
            counter += 1
            if not success:
                if counter >= frame_count:
                    break
                else:
                    errors += 1
        print(vid, "frame errors:", errors)
        video.release()
