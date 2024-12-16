from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import xml.etree.ElementTree as ET
import torchvision.transforms as T
from torch.utils.data import DataLoader

model_path = "/home/phil/university/thesis/IsaacLab-1.1.0/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/thesis-isaaclab-direct-cubechaser-camera/finetune/cube_detection_faster_rcnn"


class CubeDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        # plt.imshow(image)
        # plt.show()
        # exit()
        image_width, image_height = image.size
        image_width = float(image_width)
        image_height = float(image_height)
        annotation_path = os.path.join(self.annotation_dir,
                                       self.image_files[idx].replace('.png', '.xml'))  # Adjust if needed
        boxes = []
        labels = []
        area = []
        if os.path.exists(annotation_path):  # Check if annotation file exists
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                label = obj.find('name').text
                bbox = obj.find('bndbox')
                # Faster R-CNN works with absolute pixel coordinates for bounding boxes, not normalized coordinates.
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)
                # Assuming 'red_cube' is class 1. Adjust if you have multiple classes.
                # getting the areas of the boxes
                area.append((xmax - xmin) * (ymax - ymin))

        if len(boxes) == 0:
            # If no annotations, create entry which covers the whole image ?
            # boxes.append([0.0, 0.0, 1.0, 1.0])
            # labels.append(0)
            # or: No annotations means no cube, so empty lists for boxes and labels
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            # labels = torch.zeros((0,), dtype=torch.int64)
            labels.append(0)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = torch.as_tensor(area, dtype=torch.float32)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "area": area, "iscrowd": iscrowd}

        if self.transform:
            # image, target = self.transform(image, target)
            image = self.transform(image)

        # image = T.ToTensor()(image)  # done inside transform


        # return image, labels[0], boxes[0]
        return image, target


from tqdm import tqdm
'''
Function to train the model over one epoch.
'''
def train_one_epoch(model, optimizer, data_loader, device, epoch):
  train_loss_list = []

  tqdm_bar = tqdm(data_loader, total=len(data_loader))
  for idx, data in enumerate(tqdm_bar):
    optimizer.zero_grad()
    images, targets = data

    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # targets = {'boxes'=tensor, 'labels'=tensor}

    losses = model(images, targets)

    loss = sum(loss for loss in losses.values())
    loss_val = loss.item()
    train_loss_list.append(loss.detach().cpu().numpy())

    loss.backward()
    optimizer.step()

    tqdm_bar.set_description(desc=f"Training Loss: {loss:.3f}")

  return train_loss_list

'''
Function to validate the model
'''
def evaluate(model, data_loader_test, device):
    val_loss_list = []

    tqdm_bar = tqdm(data_loader_test, total=len(data_loader_test))

    for i, data in enumerate(tqdm_bar):
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            losses = model(images, targets)

        loss = sum(loss for loss in losses.values())
        loss_val = loss.item()
        val_loss_list.append(loss_val)

        tqdm_bar.set_description(desc=f"Validation Loss: {loss:.4f}")
    return val_loss_list

from matplotlib import pyplot as plt

def plot_loss(train_loss, valid_loss):
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()

    train_ax.plot(train_loss, color='blue')
    train_ax.set_xlabel('Iteration')
    train_ax.set_ylabel('Training Loss')

    valid_ax.plot(valid_loss, color='red')
    valid_ax.set_xlabel('Iteration')
    valid_ax.set_ylabel('Validation loss')

    figure_1.savefig(f"{model_path}/train_loss.png")
    figure_2.savefig(f"{model_path}/valid_loss.png")

    plt.close(figure_1)
    plt.close(figure_2)


# transformations
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
transforms = T.Compose([
    # for now use images in 320x240 resolution
    # T.Resize(224),  # happens anyway by the model? https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights
    # T.CenterCrop(224),  # happens anyway by the model. But make center crop equal to resize to not loose any information

    # T.RandomHorizontalFlip(),  # DONT, will mess up the bounding boxes
    # T.RandomRotation(15),  # DONT, will mess up the bounding boxes
    T.ColorJitter(brightness=0.6, contrast=0.4, saturation=0.4, hue=0.2),
    T.ToTensor(),
    # T.Normalize(mean=MEAN, std=STD)  # not needed, done by the model
])


def collate_fn(batch):
  return tuple(zip(*batch))

batch_size = 16

dataset = CubeDetectionDataset(image_dir='annotated/images', annotation_dir='annotated/Annotations', transform=transforms)
# randomly split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
  train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
  collate_fn=collate_fn)

data_loader_test = torch.utils.data.DataLoader(
  val_dataset, batch_size=1, shuffle=False, num_workers=2,
  collate_fn=collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

num_classes = 2

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1').to(device)
# should define min_size and max_size, else uses to much GPU memory during training
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT', pretrained=True, min_size=240, max_size=360)  # min_size=240, max_size=360, pretrained=True
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT', pretrained=True, min_size=240, max_size=360)  # min_size=240, max_size=360, pretrained=True

in_features = model.roi_heads.box_predictor.cls_score.in_features
print("infeatures: ", in_features)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
num_epochs = 25
lr_sched_step_size = num_epochs//2  # 3
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=lr_sched_step_size,
                                               gamma=0.1)

start_epoch = 0
loss_dict = {'train_loss': [], 'valid_loss': []}
import pickle

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    # training for one epoch
    train_loss_list = train_one_epoch(model, optimizer, data_loader, device, epoch)
    loss_dict['train_loss'].extend(train_loss_list)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    valid_loss_list = evaluate(model, data_loader_test, device)
    loss_dict['valid_loss'].extend(valid_loss_list)
    # Svae the model ckpt after every epoch
    # ckpt_file_name = f"{model_path}/epoch_{epoch + 1}_model.pth"
    # torch.save({
    #     'epoch': epoch + 1,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss_dict': loss_dict
    # }, ckpt_file_name)

    # NOTE: The losses are accumulated over all iterations
    plot_loss(loss_dict['train_loss'], loss_dict['valid_loss'])

# Store the losses after the training in a pickle
with open(f"{model_path}/loss_dict.pkl", "wb") as file:
    pickle.dump(loss_dict, file)

torch.save(model, model_path + "/fasterrcnn_cube_detector_mobilenet.pth")

print("Training Finished !")