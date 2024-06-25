from ImageClassifier.entity.config_entity import DataIngestionConfig
import zipfile
import os
import shutil
import numpy as np

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def split_data(self):

        data_dir = self.config.unzip_dir
        train_dir = self.config.train_dir
        val_dir = self.config.val_dir
        test_dir = self.config.test_dir

        for directory in [train_dir, val_dir, test_dir]:
            for class_dir in os.listdir(data_dir):
                class_path = os.path.join(data_dir, class_dir)
                if os.path.isdir(class_path):
                    files = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
                    np.random.shuffle(files)
                    train_split = int(0.7 * len(files))
                    val_split = int(0.85 * len(files))
                    self._move_files(files[:train_split], os.path.join(train_dir, class_dir))
                    self._move_files(files[train_split:val_split], os.path.join(val_dir, class_dir))
                    self._move_files(files[val_split:], os.path.join(test_dir, class_dir))
    
    def _move_files(self, files, dest):
        """
        Move specified files to a destination directory, creating the directory if it does not exist
        """
        os.makedirs(dest, exist_ok=True)
        for f in files:
            shutil.move(f, dest)
