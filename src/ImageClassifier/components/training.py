from ImageClassifier.entity.config_entity import TrainingConfig, CallbacksConfig, BaseModelConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ImageClassifier.components.base_model import BaseModel
import torch
from ImageClassifier.components.callbacks import Callback
import os
from tqdm import tqdm

class Training:
    def __init__(self, training_config: TrainingConfig, base_model_config: BaseModelConfig, callback_config: CallbacksConfig):
        self.training_config = training_config
        self.callback = Callback(callback_config)
        self.base_model = BaseModel(base_model_config)
        self.device = self.base_model.device
        self.base_model.get_base_model()
        _ = self.base_model.update_base_model()
        model, self.optimizer, self.criterion = self.base_model.full_model, self.base_model.optimizer, self.base_model.criterion
        self.model = model
        self.train_loader = None
        self.valid_loader = None

    def prepare_data(self):
        valid_transforms = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.training_config.params_is_augmentation:
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
        train_dataset = datasets.ImageFolder(root=os.path.join(self.training_config.data_path, 'train'), transform=train_transforms)
        valid_dataset = datasets.ImageFolder(root= os.path.join(self.training_config.data_path, 'val'), transform=valid_transforms)

        # Data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.training_config.params_batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.training_config.params_batch_size, shuffle=False)


    def train(self):
        self.model.train()
        self.callback.start_tensorboard_logging()
        print_every = 10
        for epoch in range(self.training_config.params_epochs):
            total_loss = 0
            total_correct = 0
            total_images = 0

            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_images += labels.size(0)
                if (i + 1) % print_every == 0:
                    print(f'Epoch {epoch+1}, Batch {i+1}, Current Loss: {loss.item()}')

            # Log metrics to TensorBoard
            avg_loss = total_loss / len(self.train_loader)
            accuracy = total_correct / total_images
            loss_val, acc_val = self.validate(self.valid_loader)
            self.callback.log_metrics({
                'Loss_train': avg_loss,
                'Accuracy_train': accuracy,
                'Loss_val': loss_val,
                'Accuracy_val': acc_val
            }, epoch)
            self.callback.save_model_checkpoint(self.model, self.optimizer, epoch, is_best=(epoch == self.training_config.params_epochs-1))

            print(f'Epoch {epoch+1}, Loss_Train: {avg_loss}, Accuracy_train: {accuracy * 100}%, Loss_val: {loss_val}, Accuracy_val: {acc_val *100}%')
        BaseModel.save_model(self.training_config.trained_model_path, self.model)
        self.callback.close()

    def validate(self, data_loader):
        self.model.eval()
        total_loss, total_correct, total_images = 0, 0, 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.base_model.device), labels.to(self.base_model.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_images += labels.size(0)
        self.model.train()
        return total_loss / len(data_loader), total_correct / total_images
