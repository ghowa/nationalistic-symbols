# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
# for mask conversion
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import cascaded_union

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class VisualizationDemo(object):

    ann_no = 1

    def create_mask_annotation(self, mask, image_id, category_id, score):
        # Find contours (boundary lines) around each sub-mask
        # Note: there could be multiple contours if the object
        # is partially occluded. (E.g. an elephant behind a tree)
        contours = measure.find_contours(mask, 0.5, positive_orientation='low')

        segmentations = []
        polygons = []
        for contour in contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            # Make a polygon and simplify it
            try:
                poly = Polygon(contour)
                poly = poly.simplify(0.05, preserve_topology=False)
                if not isinstance(poly, MultiPolygon):
                    poly = MultiPolygon([poly])
                polygons.append(poly)
                for p in poly:
                    segmentation = np.array(p.exterior.coords).ravel().tolist()
                    segmentations.append(segmentation)
            except ValueError:
                print("Wrong poly:")
                print(contour)
                continue

        # Combine the polygons to calculate the bounding box and area
        # multi_poly = MultiPolygon(polygons)
        polygons = cascaded_union(polygons)
        try:
            x, y, max_x, max_y = polygons.bounds
            width = max_x - x
            height = max_y - y
            bbox = (x, y, width, height)
            area = polygons.area

            annotation = {
                'id': self.ann_no,
                'segmentation': segmentations,
                'image_id': image_id,
                'category_id': int(category_id),
                'score': float(score),
                'bbox': bbox,
                'area': float(area)
            }
            self.ann_no += 1
            return annotation
        except ValueError:
            print("Wrong poly:")
            print(polygons)
            return None

    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        counter = 0
        while video.isOpened():
            success, frame = video.read()
            counter += 1
            if success:
                yield frame
            else:
                if counter >= frame_count:
                    break
                else:
                    print("Faulty frame: ", counter)

    def run_on_video(self, video, frame_skip=0, visualization=False):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
            frame_skip: an integer indicating how many frames should be skipped during inference

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions, frame_no):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            annotations = []
            vis_frame = None
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                if visualization:
                    vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                        frame, panoptic_seg.to(self.cpu_device), segments_info
                    )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                if visualization:
                    vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
                # save instances to file
                for index in range(len(predictions)):
                    pred_mask = predictions.get('pred_masks').numpy()[index]
                    category_id = predictions.get('pred_classes').numpy()[index]
                    score = predictions.get('scores').numpy()[index]
                    anns = self.create_mask_annotation(pred_mask, frame_no, category_id, score)
                    if anns is not None:
                        annotations.append(anns)

            elif "sem_seg" in predictions:
                if visualization:
                    vis_frame = video_visualizer.draw_sem_seg(
                        frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                    )

            # Converts Matplotlib RGB format to OpenCV BGR format
            if visualization:
                vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame, annotations

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            frame_no = 0
            for frame in frame_gen:
                frame_no += 1
                if frame_no % frame_skip == 0:
                    yield process_predictions(frame, self.predictor(frame), frame_no)


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
