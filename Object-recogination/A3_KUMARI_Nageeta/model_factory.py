"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms, vit_data_transforms, dinov2_data_transforms, vit_updated_transform, dinov2_data_transforms_extended
from model import Net, ExtendedNet
from transformers import ViTForImageClassification, ViTImageProcessor
import torchvision.transforms as transforms
import torch
import torch.nn as nn


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        def load_and_modify_dino_model(model_name, num_classes=500):
            model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
            # freeze backbone
            for param in model.parameters():
                param.requires_grad = False
            # craete new head
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                features = model(dummy_input)
            feature_dim = features.shape[-1]
            model.head = nn.Linear(feature_dim, num_classes)
            # unfreeze head
            for param in model.head.parameters():
                param.requires_grad = True
            return model
        
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "extended_cnn":
            return ExtendedNet()
        elif self.model_name == "vit":
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, 500)
            # Freeze the backbone and unfreeze classifier
            for param in model.vit.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
            return model
        elif self.model_name in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitb14_reg']:
            return load_and_modify_dino_model(self.model_name)
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented.")


    def init_transform(self):
        if self.model_name == "vit":
            # return vit_data_transforms
            return vit_updated_transform
        
        elif self.model_name == "basic_cnn" or self.model == 'extended_cnn':
            return data_transforms
        
        elif self.model_name in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitb14_reg']:
            # return dinov2_data_transforms
            return dinov2_data_transforms_extended
        
        else:
            raise NotImplementedError("Transform not implemented")



    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
