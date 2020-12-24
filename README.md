# Yolo v3 - Demo Application (AIHub - Dangerous Object Image Dataset)

![demo_application](https://github.com/ByeongGil-Jung/Yolo_v3-Dangerous-Object-Image-Demo-Application/blob/master/github/img/thumbnail.png)

## Introduction
Yolo v3 를 활용한 데모 어플리케이션입니다.  
데이터셋은 AIhub 에서 제공한 위해물품 엑스레이 이미지를 사용하였으며,  
백엔드로 Flask 를 사용하였습니다.  
  
왼쪽의 view 버튼을 통해 판정하고자 하는 이미지를 확인할 수 있고, inferecne 버튼을 통해 판정을 수행할 수 있습니다.  
판정 결과의 예시는 위 그림과 같습니다.  
(Yolo v3 의 구현체로는 [Erik Linder-Noren](https://github.com/eriklindernoren) 의  [여기](https://github.com/eriklindernoren/PyTorch-YOLOv3) 를 활용하였습니다.)

## Prerequisite
#### - 사전 학습된 모델을 그대로 사용하고자 할 경우
1. 사전 학습된 모델을 사용하고 싶을 경우, `/model/yolov3/checkpoints` 폴더의 `yolov3_ckpt.zip` 의 압축을 풉니다.
2. 추가로 판정을 원하는 이미지가 있을 경우, `/model/yolov3/data/samples` 에 해당 이미지들을 삽입합니다.  

> 해당 모델은 Astrophysics 머신의 데이터셋만 사용하였습니다.  
> 그 중 20개의 클래스를 추출하여 총 38854 개의 데이터를 활용하였고,  
> 150 번의 epoch 을 수행하였습니다.  
> 학습에 사용된 클래스는 아래와 같습니다.
```
Axe
Bat
Battery
Gun
Hammer
HandCuffs
HDD
Knife
Laptop
Lighter
Saw
Scissors
Screwdriver
SmartPhone
Spanner
SupplymentaryBattery
Throwing Knife
USB
Plier
Chisel
```

#### - 새로 학습한 모델을 사용하고자 할 경우
1. 판정을 위한 모델을 `/model/yolov3/checkpoints` 로 옮긴 뒤, 파일 명을 `yolov3_ckpt.pth` 로 바꿉니다.
2. 학습을 수행했던 모듈의 `/config` , `/data/custom` 폴더를 현재 모듈의 `/model/yolov3/config` , `/model/yolov3/data/custom` 폴더와 바꿉니다.
3. 판정을 원하는 이미지들을 `/model/yolov3/data/samples` 에 삽입합니다.
4. 추가로 세부적으로 파라미터를 조정하고자 할 경우, `/properties.py` 를 참조하여 수정합니다.

Model 의 API 를 사용하고자한다면 아래와 같이 사용하도록 합니다.
```
model_api = ModelAPI(
    image_folder=SAMPLE_DIRECTORY_PATH,
    model_def=MODEL_CONFIG_FILE_PATH,
    weights_path=MODEL_CHECKPOINTS_DIRECTORY_PATH,
    class_path=CLASS_LIST_FILE_PATH,
    conf_thres=CONFIDENCE_THRESHOLD,
    nms_thres=NON_MAX_SUPPRESSION_THRESHOLD,
    batch_size=BATCH_SIZE,
    n_cpu=N_CPU,
    img_size=IMG_SIZE,
    device=DEVICE
)
```

## How to run
(1) 파이썬 디펜던시 설치
```
pip3 install -r ./requirements.txt
```
(2) Flask 웹 어플리케이션 실행
```
python3 ./app.py
```
(3) 접근  
(Default port : 5000)
```
http://localhost:5000
```

---
Thanks to [Erik Linder-Noren](https://github.com/eriklindernoren).  
([Original Yolo_v3 Module Repository](https://github.com/eriklindernoren/PyTorch-YOLOv3))
