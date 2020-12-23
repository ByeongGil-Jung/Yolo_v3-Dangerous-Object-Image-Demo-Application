import base64
import gc
import json
import io

from flask import Flask, render_template, request, Response

from api import ModelAPI
from logger import logger
from model.yolov3.utils.datasets import *
from properties import APPLICATION_PROPERTIES

app = Flask(__name__)


model_api = ModelAPI(
    image_folder=os.path.join(APPLICATION_PROPERTIES.YOLO_MODULE_PATH, "data", "samples"),
    model_def=os.path.join(APPLICATION_PROPERTIES.YOLO_MODULE_PATH, "config", "yolov3-custom.cfg"),
    weights_path=os.path.join(APPLICATION_PROPERTIES.YOLO_MODULE_PATH, "checkpoints", "yolov3_ckpt_150.pth"),
    class_path=os.path.join(APPLICATION_PROPERTIES.YOLO_MODULE_PATH, "data", "custom", "classes.names"),
    conf_thres=0.8,
    nms_thres=0.4,
    batch_size=1,
    n_cpu=0,
    img_size=416,
    device=ModelAPI.DEVICE_GPU
)


@app.route('/')
def index():
    sample_dir_path = APPLICATION_PROPERTIES.SAMPLE_DIRECTORY_PATH
    sample_img_file_path_list = os.listdir(sample_dir_path)

    sample_img_filename_list = list(map(lambda img_file_path: os.path.basename(img_file_path), sample_img_file_path_list))

    return render_template(
        "index.html",
        sample_img_filename_list=sample_img_filename_list
    )


@app.route("/inference", methods=["POST"])
def inference():
    sample_img_filename = request.form["sample_img_filename"]
    sample_img_file_path = os.path.join(APPLICATION_PROPERTIES.SAMPLE_DIRECTORY_PATH, sample_img_filename)
    result_text = ""

    # Do inference
    result_dict = model_api.detect(img_file_path=sample_img_file_path)

    inference_img_file_path = os.path.join(APPLICATION_PROPERTIES.INFERENCE_DIRECTORY_PATH, sample_img_filename)
    encoded_img = get_base64_encoded_img(img_file_path=inference_img_file_path)

    # Set result text
    for label, confidence in zip(result_dict["label_list"], result_dict["confidence_list"]):
        result_text += f"{label} : {confidence}\n"

    logger.info(f"Success to inference the image : {sample_img_file_path}")

    response_json = json.dumps(dict(
        inference_img_file_name=sample_img_filename,
        inference_img_file_path=inference_img_file_path,
        inference_base64_encoded_img=encoded_img,
        result_text=result_text
    ))

    return Response(response_json, status=200, mimetype="application/json")


@app.route("/view", methods=["POST"])
def view():
    sample_img_filename = request.form["sample_img_filename"]
    view_img_file_path = os.path.join(APPLICATION_PROPERTIES.SAMPLE_DIRECTORY_PATH, sample_img_filename)
    encoded_img = get_base64_encoded_img(img_file_path=view_img_file_path)

    logger.info(f"Success to load the view image : {view_img_file_path}")

    response_json = json.dumps(dict(
        view_img_file_name=sample_img_filename,
        view_img_file_path=view_img_file_path,
        view_base64_encoded_img=encoded_img
    ))
    return Response(response_json, status=200, mimetype="application/json")


@app.after_request
def after_request(response):
    gc.collect()  # Collecting garbage

    return response


def get_base64_encoded_img(img_file_path):
    img = Image.open(img_file_path, mode="r")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="png")
    encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode("ascii")

    return encoded_img


def load_model():
    model_api.load_model()


def activate_app():
    load_model()


if __name__ == '__main__':
    activate_app()
    app.run(debug=True, port=5000)
