import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from ImageClassifier.entity.config_entity import CallbacksConfig
from pathlib import Path

class Callback:
    def __init__(self, config: CallbacksConfig):
        self.config = config
        self.writer = None  # TensorBoard writer
        
    def start_tensorboard_logging(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}"
        )
        self.writer = SummaryWriter(log_dir=tb_running_log_dir)
        
    def log_metrics(self, metric_dict, step):
        if self.writer:
            for tag, value in metric_dict.items():
                self.writer.add_scalar(tag, value, step)
    
    def save_model_checkpoint(self, model, optimizer, epoch, is_best=False):
        checkpoint_path = self.config.checkpoint_model_filepath
        if is_best:
            if not isinstance(checkpoint_path, str):
                checkpoint_path = str(checkpoint_path)
            checkpoint_path = checkpoint_path.replace('.pth', f'_best_epoch_{epoch}.pth')
            checkpoint_path = Path(checkpoint_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
    
    def close(self):
        if self.writer:
            self.writer.close()


