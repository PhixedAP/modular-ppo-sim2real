from torchvision import transforms
import mimetypes
import argparse
import pickle
import torch
import cv2
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
# from object_detector import ObjectDetector
import os
import random
import torchvision

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
sample_image_path = "/home/phil/university/thesis/IsaacLab-1.1.0/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/thesis-isaaclab-direct-cubechaser-camera/finetune/cube_detection_faster_rcnn/annotated/images/"
# sample_image_path = "/home/phil/university/thesis/thesis-ros-jetauto/local_ws/src/fourwis_cubechaser_sb3/scripts/cube_detector/sample_images/"
# sample_image_path = "/home/phil/university/thesis/thesis-ros-jetauto/local_ws/src/fourwis_cubechaser_sb3/scripts/cube_detector/"

model_path = "/home/phil/university/thesis/IsaacLab-1.1.0/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/thesis-isaaclab-direct-cubechaser-camera/finetune/cube_detection_faster_rcnn/"
model = torch.load(model_path + "fasterrcnn_cube_detector_mobilenet.pth").to(device)
model.eval()

transforms = T.Compose([
    T.ToTensor(),
])


def apply_nms(orig_prediction, iou_thresh=0.5):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    print("keep:", keep)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction

while True:
    # randomly select an image from sample image path
    sample_image_name = random.choice(os.listdir(sample_image_path))
    # sample_image_name = "test.jpg"
    print("sample_image_name:", sample_image_name)

    image = Image.open(sample_image_path + sample_image_name).convert("RGB")
    image_width, image_height = image.size
    print("image:", image.size)
    image = transforms(image).unsqueeze(0).to(device)


    predction = model(image)
    print("predction:", predction)
    final_pred = apply_nms(predction[0])
    print("final_pred:", final_pred)
    boxPreds = final_pred['boxes'].cpu().detach().numpy()
    classPreds = final_pred['labels'].cpu().detach().numpy()
    print("boxPreds:", boxPreds)
    print("classPreds:", classPreds)

    image_cv2 = cv2.imread(sample_image_path + sample_image_name)
    if len(boxPreds) > 0:
        startX = int(boxPreds[0][0])
        startY = int(boxPreds[0][1])
        endX = int(boxPreds[0][2])
        endY = int(boxPreds[0][3])

        print(f"startX: {startX}, startY: {startY}, endX: {endX}, endY: {endY}")

        cv2.rectangle(image_cv2, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.imshow("image", image_cv2)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == ord('q'):
        break
