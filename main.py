import os
import torch

import albumentations
import pretrainedmodels

import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn import metrics
from torch.nn import functional as F

from wtfml.data_loaders.image import ClassificationLoader
from wtfml.engine import Engine
from wtfml.utils import EarlyStopping


class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        self.model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=pretrained)
        self.out = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.reshape(-1, 1).type_as(out))
        return out, loss

def train(fold):
    training_data_path = "E:/documents/dev/skin cancer detection data/siim-isic-melanoma-classification/jpeg/train224"
    model_path = "C:/Users/sarab/OneDrive/Documents/dev/skin cancer detection app/model.bin"
    df = pd.read_csv("E:/documents/dev/skin cancer detection data/siim-isic-melanoma-classification/train_folds.csv")
    device = "cuda"
    epochs = 50
    train_bs = 32
    valid_bs = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=512.0, always_apply=True),
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".jpg") for i in valid_images]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=False,
        num_workers=4
    )

    valid_dataset = ClassificationLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_bs,
        shuffle=False,
        num_workers=4
    )

    model = SEResNext50_32x4d(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode="max"
    )

    es = EarlyStopping(patience=5, mode="max")
    for epoch in range(epochs):
        training_loss = Engine.train(
            train_loader,
            model,
            optimizer,
            device,
            fp16=False
        )
        predictions, valid_loss = Engine.evaluate(
            valid_loader,
            model,
            device
        )
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)
        print(f"epoch={epoch}, auc={auc}")
        es(auc, model, model_path)
        if es.early_stop:
            print("early stopping")
            break

if __name__ == "__main__":
    train(fold=0)