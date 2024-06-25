import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from ImageClassifier.entity.config_entity import BaseModelConfig


class BaseModel:
    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_base_model(self):
        # Load pretrained Inception V3
        if self.config.params_weights == 'imagenet':
            model = models.inception_v3(pretrained=True)
        else:
            model = models.inception_v3(pretrained=False)
        
        if not self.config.params_include_top:
            model = nn.Sequential(*list(model.children())[:-1])
        
        self.model = model.to(self.device)
        self.save_model(path=self.config.base_model_path, model=self.model)

    def _prepare_full_model(self, model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif freeze_till is not None and freeze_till > 0:
            children = list(model.children())
            for i in range(min(freeze_till, len(children))):
                for param in children[i].parameters():
                    param.requires_grad = False

        model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, classes),
            nn.Softmax(dim=1)
        ).to(self.device)

        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        return model, optimizer, criterion

    def update_base_model(self):
        self.full_model, self.optimizer, self.criterion = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path, model):
        torch.save(model.state_dict(), path)


