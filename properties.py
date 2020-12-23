from dataclasses import dataclass
import os
import pathlib
import random

import numpy as np
import torch

from logger import logger


class Configuration(object):

    DEFAULT_RANDOM_SEED = 777

    @classmethod
    def apply(cls, random_seed=DEFAULT_RANDOM_SEED):
        Configuration.set_torch_seed(random_seed=random_seed)
        Configuration.set_numpy_seed(random_seed=random_seed)
        Configuration.set_python_random_seed(random_seed=random_seed)

        logger.info(f"Complete to apply the random seed, RANDOM_SEED : {random_seed}")

    @classmethod
    def set_torch_seed(cls, random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def set_numpy_seed(cls, random_seed):
        np.random.seed(random_seed)

    @classmethod
    def set_python_random_seed(cls, random_seed):
        random.seed(random_seed)


@dataclass
class ApplicationProperties:
    CURRENT_MODULE_PATH = pathlib.Path(__file__).parent.absolute()

    MODEL_DIRECTORY_PATH = os.path.join(CURRENT_MODULE_PATH, "model")
    YOLO_MODULE_PATH = os.path.join(MODEL_DIRECTORY_PATH, "yolov3")

    SAMPLE_DIRECTORY_PATH = os.path.join(YOLO_MODULE_PATH, "data", "samples")
    INFERENCE_SAMPLE_DIRECTORY_PATH = os.path.join(YOLO_MODULE_PATH, "data", "inference_samples")
    INFERENCE_DIRECTORY_PATH = os.path.join(YOLO_MODULE_PATH, "output")

    MODEL_CONFIG_FILE_PATH = os.path.join(YOLO_MODULE_PATH, "config", "yolov3-custom.cfg")
    MODEL_CHECKPOINTS_DIRECTORY_PATH = os.path.join(YOLO_MODULE_PATH, "checkpoints", "yolov3_ckpt.pth")
    CLASS_LIST_FILE_PATH = os.path.join(YOLO_MODULE_PATH, "data", "custom", "classes.names")

    DEFAULT_RANDOM_SEED = 777

    DEVICE_CPU = "cpu"

    # Inference parameters
    CONFIDENCE_THRESHOLD = 0.8
    NON_MAX_SUPPRESSION_THRESHOLD = 0.4
    BATCH_SIZE = 1
    N_CPU = 0
    IMG_SIZE = 416

    def __post_init__(self):
        Configuration.apply(random_seed=self.DEFAULT_RANDOM_SEED)


APPLICATION_PROPERTIES = ApplicationProperties()
