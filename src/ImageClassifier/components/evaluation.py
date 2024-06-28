import torch
from ImageClassifier.entity.config_entity import EvaluationConfig, BaseModelConfig
from pathlib import Path
from torch.utils.data import DataLoader
from ImageClassifier.utils.common import save_json
from torchvision import datasets, transforms
import os
from ImageClassifier.components.base_model import BaseModel

class Evaluation:
    def __init__(self, config_eval: EvaluationConfig, config_model: BaseModelConfig):
        self.config_eval = config_eval
        self.config_model = config_model
        base_model = BaseModel(config=config_model)
        self.device = base_model.device
        base_model.get_base_model()
        self.model = base_model.update_base_model()
    
    def _test_generator(self):

        test_transforms = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_dataset = datasets.ImageFolder(root= os.path.join(self.config_eval.test_dir), transform=test_transforms)
        self.test_generator = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0)


    
    def load_model(self) -> torch.nn.Module:
        state_dict = torch.load(self.config_eval.trained_model_path)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    def evaluation(self):
        self._test_generator()
        self.load_model()
        total_loss = 0
        total_correct = 0
        total_images = 0
        criterion = torch.nn.CrossEntropyLoss()
        print_every = 100
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_generator):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_images += inputs.size(0)
                if (i + 1) % print_every == 0:
                    print(f'Batch {i+1}, Current Loss: {loss.item()},  Current Acc: {total_correct / total_images * 100}')

        avg_loss = total_loss / total_images
        accuracy = total_correct / total_images * 100
        self.score = (avg_loss, accuracy)

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
