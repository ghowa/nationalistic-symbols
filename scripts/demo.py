# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances

from predictor import VisualizationDemo

import json

# constants
WINDOW_NAME = "COCO detections"

metadata = {
    "thing_classes": [
        "adler",
        "bandera1",
        "bandera2",
        "bandera3",
        "eu",
        "falanga",
        "flag_ru_hang",
        "flag_ru_fly",
        "flag_ru",
        "flag_ua_hang",
        "flag_ua_fly",
        "flag_ua",
        "flag_upa_hang",
        "flag_upa_fly",
        "flag_upa",
        "george_hang",
        "george_fly",
        "george_band",
        "george",
        "swastika",
        "hammer",
        "cross",
        "orthodox",
        "nato",
        "oun",
        "ss",
        "swoboda",
        "ukraine",
        "wolfsangel",
    ]
}


def setup_cfg(args):
    # load config from file and command-line arguments
    # next line is only needed to get class names; test.json and test_images do not need to exist as files/directories!
    register_coco_instances("bandera_test", metadata,
                            "test.json", "test_images")
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.DATASETS.TEST = ("bandera_test",)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Skip frames to speed up the process",
    )
    # TODO: Add --no-vis
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    vis_demo = VisualizationDemo(cfg)

    if args.video_input:
        vid_list = []
        path = ""
        if os.path.isdir(args.video_input):
            vid_list = os.listdir(args.video_input)
            path = args.video_input + os.path.sep 
        elif os.path.isfile(args.video_input):
            vid_list = [args.video_input]

        for vid in vid_list:
            video = cv2.VideoCapture(os.path.join(path, vid))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            basename = os.path.basename(vid)

            output_dict = {
                'frame_skip': args.frame_skip,
                'width': width,
                'height': height,
                'fps': frames_per_second,
                'frames': num_frames,
                'file': vid,
                'annotations': []
            }

            if args.output:
                if os.path.isdir(args.output):
                    output_fname = os.path.join(args.output, basename)
                    output_fname = os.path.splitext(output_fname)[0] + ".mp4"
                    vid_json = output_fname + ".json"
                else:
                    output_fname = args.output
                assert not os.path.isfile(output_fname), output_fname
                output_file = cv2.VideoWriter(
                    filename=output_fname,
                    fourcc=cv2.VideoWriter_fourcc(*"MPEG"),
                    fps=float(frames_per_second),
                    frameSize=(width, height),
                    isColor=True,
                )
                print(vid)
            assert os.path.isfile(os.path.join(path, vid))
            for vis_frame, annotations in tqdm.tqdm(vis_demo.run_on_video(video, args.frame_skip, visualization=True), total=num_frames / args.frame_skip):
                if len(annotations) > 0 and args.output:
                    output_dict['annotations'].extend(annotations)
                if vis_frame is not None:
                    if args.output:
                        for i in range(args.frame_skip):
                            output_file.write(vis_frame)
                    else:
                        cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                        cv2.imshow(basename, vis_frame)
                        if cv2.waitKey(1) == 27:
                            break  # esc to quit
            video.release()

            if args.output:
                output_file.release()
                with open(vid_json, 'w') as output_json:
                    json.dump(output_dict, output_json, indent=4)
            else:
                cv2.destroyAllWindows()
