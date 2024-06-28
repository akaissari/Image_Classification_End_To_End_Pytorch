import numpy as np
from ImageClassifier.constants import *
import os
from pathlib import Path
from ImageClassifier.components.base_model import BaseModel
import torch
from PIL import Image
from torchvision import transforms
import json
from ImageClassifier.config.configuration import ConfigurationManager
from idx_to_class import idx_to_class

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename
        config = ConfigurationManager()
        test_config = config.get_evaluation_config()
        config_model = config.get_base_model_config()

        self.path_of_model = test_config.trained_model_path
        base_model = BaseModel(config=config_model)
        self.device = base_model.device
        base_model.get_base_model()
        self.model = base_model.update_base_model()
        state_dict = torch.load(self.path_of_model)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode



    def prepare_image(self):
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        imagename = self.filename
        image = Image.open(imagename).convert('RGB')
        input_tensor = transform(image)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        return input_tensor
        
    def predict(self):
        input_tensor = self.prepare_image()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        print(probabilities)
        class_predicted = idx_to_class[str(torch.argmax(probabilities).item())]
        return [{ "image" : class_predicted}]