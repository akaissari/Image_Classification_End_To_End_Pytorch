import torch
from ImageClassifier.entity.config_entity import EvaluationConfig
from pathlib import Path
from torch.utils.data import DataLoader
from ImageClassifier.utils.common import save_json
from torchvision import datasets, transforms
import os

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _test_generator(self):

        test_transforms = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_dataset = datasets.ImageFolder(root= os.path.join(self.config.test_dir), transform=test_transforms)
        self.test_generator = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=-1)


    
    @staticmethod
    def load_model(path: Path) -> torch.nn.Module:
        model = torch.load(path)
        model.eval()  # Set the model to evaluation mode
        return model

    def evaluation(self):
        self._test_generator()
        self.model.to(self.device)
        total_loss = 0
        total_correct = 0
        total_images = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in self.test_generator:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_images += inputs.size(0)

        avg_loss = total_loss / total_images
        accuracy = total_correct / total_images * 100
        self.score = (avg_loss, accuracy)

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
