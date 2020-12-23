from shutil import copyfile
import datetime
import os
import pathlib
import random
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import patches
from matplotlib.ticker import NullLocator
from torch.autograd import Variable
from torch.utils.data import DataLoader

from logger import logger
from model.yolov3.models import Darknet
from model.yolov3.utils.datasets import ImageFolder
from model.yolov3.utils.utils import load_classes, non_max_suppression, rescale_boxes
from properties import APPLICATION_PROPERTIES


class ModelAPI(object):

    DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE_CPU = torch.device("cpu")

    def __init__(self, image_folder, model_def, weights_path, class_path, conf_thres, nms_thres, batch_size, n_cpu, img_size, device):
        self.image_folder = image_folder
        self.model_def = model_def
        self.weights_path = weights_path
        self.class_path = class_path
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.batch_size = batch_size
        self.n_cpu = n_cpu
        self.img_size = img_size
        self.weights_path = weights_path
        self.device = device

        self.model = Darknet(self.model_def, img_size=self.img_size).to(self.device)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.weights_path, map_location=torch.device("cpu")))
        self.model.to(self.device)

        logger.info(f"Success to load model : {self.weights_path}, device : {self.device}")

    def detect(self, img_file_path):
        logger.info("Start to inference")
        self.model.eval()

        label_list = list()
        confidence_list = list()

        save_inference_img_file_path = ""

        if os.path.isfile(img_file_path):
            img_filename = os.path.basename(img_file_path)
            save_inference_img_file_path = os.path.join(APPLICATION_PROPERTIES.INFERENCE_SAMPLE_DIRECTORY_PATH, img_filename)
            copyfile(img_file_path, save_inference_img_file_path)

        dataloader = DataLoader(
            ImageFolder(APPLICATION_PROPERTIES.INFERENCE_SAMPLE_DIRECTORY_PATH, img_size=self.img_size),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_cpu,
        )

        classes = load_classes(self.class_path)
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        imgs = []
        img_detections = []

        print("\nPerforming object detection:")
        prev_time = time.time()
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = self.model(input_imgs)
                detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

            # Save image and detections
            imgs.extend(img_paths)
            img_detections.extend(detections)

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        print("\nSaving images:")
        # Iterate through images and save plot of detections
        size_mult = 4

        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

            print("(%d) Image: '%s'" % (img_i, path))

            # Create plot
            img = np.array(Image.open(path))
            plt.figure()
            fig, ax = plt.subplots(1, figsize=(6*size_mult, 4*size_mult))
            ax.imshow(img)

            # Draw bounding boxes and labels of detections
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, self.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                    label_list.append(classes[int(cls_pred)])
                    confidence_list.append(round(cls_conf.item(), 3))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                        fontsize=20
                    )

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            sample_file_path = pathlib.Path(path)
            filename = os.path.basename(sample_file_path)

            save_path = os.path.join(APPLICATION_PROPERTIES.YOLO_MODULE_PATH, "output", f"{filename}")

            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
            plt.close()

        os.remove(save_inference_img_file_path)

        logger.info("Success to inference")

        return dict(
            label_list=label_list,
            confidence_list=confidence_list
        )
