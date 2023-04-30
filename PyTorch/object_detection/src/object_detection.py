import torch
from torch.utils.data import Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

import cv2


class LoadDataset(Dataset):
    def __init__(self, df, img_dir, transforms):
        super().__init__()
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # read & process the image
        filename = self.df.loc[idx, "image"]
        img = cv2.imread(str(self.img_dir / filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        # get the bboxes
        bboxes = self.df.loc[idx, ["xmin", "ymin", "xmax", "ymax"]].values
        bboxes = tuple(map(torch.tensor, zip(*bboxes)))
        bboxes = torch.stack(bboxes, dim=0)

        # create labels
        labels = torch.ones(len(bboxes), dtype=torch.int64)
        # apply augmentations
        augmented = self.transforms(image=img, bboxes=bboxes, labels=labels)

        # convert bbox list to tensors again
        bboxes = map(torch.tensor, zip(*augmented["bboxes"]))
        bboxes = tuple(bboxes)
        bboxes = torch.stack(bboxes, dim=0)

        img = augmented["image"].type(torch.float32)
        bboxes = bboxes.permute(1, 0).type(torch.float32)
        iscrowd = torch.zeros(len(bboxes), dtype=torch.int)

        # bbox area
        area = self.df.loc[idx, "area"]
        torch.as_tensor(area, dtype=torch.float32)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target


class ObjectDetector(LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.batch_size = 16
        self.model = self.create_model()
        self.validation_step_outputs = []  # Added by me

    def create_model(self):
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        return model

    def forward(self, x):
        return self.model(x)

    # The number of bounding boxes may be different for each image,
    # so, we will need a collate_fn to pass to our dataloaders.
    def collate_fn(self, batch):
        return tuple(zip(*batch))

    # def train_dataloader(self):
    #     return DataLoader(
    #         train_ds, 
    #         batch_size=self.batch_size, 
    #         shuffle=True, 
    #         collate_fn=self.collate_fn
    #     )

    # def val_dataloader(self):
    #     return DataLoader(
    #         val_ds, 
    #         batch_size=self.batch_size, 
    #         shuffle=False, 
    #         collate_fn=self.collate_fn
    #     )
    
    # def configure_optimizers(self):
    #     return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    # # For training, instead of giving only the predictions,
    # # the model outputs a bunch of loss values.
    # # So, we should sum all these losses together and
    # # return it in training step for backward propagation.
    # def training_step(self, batch, batch_idx):
    #     inputs, targets = batch
    #     loss_dict = self.model(inputs, targets)
    #     complete_loss = sum(loss for loss in loss_dict.values())

    #     self.log("train_loss", complete_loss, prog_bar=True)
    #     return {"loss": complete_loss}

    # # During validation, pytorch lightning automatically calls model.eval() for us.
    # # While doing this, the behaviour of the model will change again.
    # # This time, the model will output the bounding box prediction and probabilites of our label(car).
    # # So we need to take this into account while implementing the validation step.

    # # So, during validation, we take the predicted bounding box coordinates and the target bounding boxes
    # # to calculate intersection over union(IOU) which is a commonly used metric for object detection.
    # # We will be using box_iou function from torchvision for calculating IOU.

    # # IOU varies from 0 to 1, values closer to 0 are considered bad
    # # whereas the ones closer to 1 are considered good predictions.
    # def validation_step(self, batch, batch_idx):
    #     inputs, targets = batch
    #     outputs = self.model(inputs)
    #     # calculate IOU and return the mean IOU
    #     iou = torch.stack(
    #         [
    #             box_iou(target["boxes"], output["boxes"]).diag().mean()
    #             for target, output in zip(targets, outputs)
    #         ]
    #     ).mean()
    #     self.validation_step_outputs.append(iou)  # Added by me
        
    #     return {"val_iou": iou}

    # # So, from validation_step() we will get the IOU for each batch,
    # # this is appended to a list and passed to validation_epoch_end().
    # # So, the only task remaining is to calculate the mean IOU from the list
    # # passed to validation_epoch_end() and log it:
    # def on_validation_epoch_end(self):  # modified by me
    #     # calculate overall IOU across batch
    #     # val_iou = torch.stack([o["val_iou"] for o in val_out]).mean()  # Removed by me
    #     val_iou = torch.stack(self.validation_step_outputs).mean()  # Added by me
    #     self.validation_step_outputs.clear()  # free memory
    #     self.log("val_iou", val_iou, prog_bar=True)
    #     return val_iou