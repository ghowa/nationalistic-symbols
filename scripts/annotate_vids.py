#! /bin/python
import cv2
import os
import json
import sys
import numpy as np
from tqdm import tqdm
from detectron2.utils.colormap import colormap

CATS = []
with open("annotations/bandera-v4.0.0-categories") as f:
    CATS = f.read().split("\n")

def main():
    if len(sys.argv) <= 1:
        print("Usage: python annotate_vids.py $VID_FOLDER $OUTPUT_FOLDER [$OUTPUT_AS_VIDEO]")
        print("Example: python annotate_vids.py ../videos ../output true")
        exit(0)

    continuous = False
    if len(sys.argv) == 4 and sys.argv[3] == "true":
        continuous = True

    def _frame_from_video(video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    for vid in os.listdir(sys.argv[1]):
        if vid.endswith("json"):
            continue
        vid_file = os.path.join(sys.argv[1], vid)
        vid_json = os.path.join(sys.argv[1], vid + ".json")

        # load annotations
        with open(vid_json) as json_file:
            anns = json.load(json_file)
        frame_skip = anns['frame_skip']
        
        anns_per_frame = {}

        for ann in anns["annotations"]:
            try:
                anns_per_frame[ann["image_id"]].append(ann)
            except KeyError:
                anns_per_frame[ann["image_id"]]=[ann]

        # load vids
        video = cv2.VideoCapture(vid_file)
        frame_gen = _frame_from_video(video)
        frame_no = 0
        fps = video.get(cv2.CAP_PROP_FPS)
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # prepare output directory or output vid
        vid_base = os.path.join(sys.argv[2], vid.split(".")[0])
        writer = None
        if continuous:
            writer = cv2.VideoWriter(
                filename=vid_base + "_annotated.mp4",
                apiPreference=0,
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=int(fps / frame_skip),
                frameSize=(int(width), int(height)),
                isColor=True,
            )
        else:
            if not os.path.isdir(vid_base):
                os.mkdir(vid_base)

        with tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT))) as bar:
            for frame in frame_gen:
                bar.update(1)
                frame_no += 1
                result = frame.copy()
                if frame_no in anns_per_frame.keys():
                    for ann in anns_per_frame[frame_no]:

                        # draw ann on frame
                        cat = ann['category_id']
                        score = ann['score']
                        poly = ann['segmentation']
                        box = ann['bbox']

                        # convert and draw poly
                        for p in poly:
                            overlay = frame.copy()
                            p = np.array(p)
                            p1 = []
                            for index, point in enumerate(p):
                                if index % 2 == 0:
                                    continue
                                else:
                                    p1.append([p[index - 1], p[index]])

                            p1 = np.int32([p1])

                            c = tuple(map(int, colormap()[cat]))
                            cv2.fillPoly(overlay, p1, c)
                            cv2.addWeighted(overlay, 0.5, result, 1 - 0.5, 0, result)

                        # add text

                        cv2.putText(result, CATS[cat] + " (" + str(round(score, 2)) + ")", (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX , 0.5, c, 1, cv2.LINE_AA)

                # write output
                if continuous:
                    writer.write(result)
                else:
                    cv2.imwrite(os.path.join(vid_base, str(ann['image_id']) + ".jpg"), result)
                    
        if continuous:
            writer.release()


if __name__ == "__main__":
    main()
