from ImageClassifier.entity.config_entity import TrainingConfig, CallbacksConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ImageClassifier.components.base_model import BaseModel
import torch
from ImageClassifier.components.callbacks import Callback
import os

class Training:
    def __init__(self, config: TrainingConfig, callback_config: CallbacksConfig):
        self.config = config
        self.callback = Callback(callback_config)
        self.base_model = BaseModel(config)
        self.train_loader = None
        self.valid_loader = None

    def prepare_data(self):
        valid_transforms = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.config.params_is_augmentation:
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=40, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=20),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        train_dataset = datasets.ImageFolder(root=os.path.join(self.config.data_path, 'train'), transform=train_transforms)
        valid_dataset = datasets.ImageFolder(root= os.path.join(self.config.data_path, 'val'), transform=valid_transforms)

        # Data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.config.params_batch_size, shuffle=False)


    def train(self):
        self.base_model.get_base_model()
        self.base_model.update_base_model()
        model, optimizer, criterion = self.base_model.full_model, self.base_model.optimizer, self.base_model.criterion
        model.train()

        # Start TensorBoard logging
        self.callback.start_tensorboard_logging()

        for epoch in range(self.config.params_epochs):
            total_loss = 0
            total_correct = 0
            total_images = 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.base_model.device), labels.to(self.base_model.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_images += labels.size(0)

            # Log metrics to TensorBoard
            avg_loss = total_loss / len(self.train_loader)
            accuracy = total_correct / total_images
            self.callback.log_metrics({
                'Loss': avg_loss,
                'Accuracy': accuracy
            }, epoch)
            self.callback.save_model_checkpoint(model, optimizer, epoch, is_best=(epoch == self.config.params_epochs-1))

            print(f'Epoch {epoch+1}, Loss: {avg_loss}, Accuracy: {accuracy * 100}%')
        BaseModel.save_model(self.config.trained_model_path, model)
        self.callback.close()