from pathlib import Path
import PIL
from torch.utils.data import Dataset


class BreastDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        patient_id = self.dataframe.patient_id.values[index]
        x_coordinate = self.dataframe.x.values[index]
        y_coordinate = self.dataframe.y.values[index]
        class_name = self.dataframe.label.values[index]
        image_path = self.dataframe.image_path.values[index]
        image = PIL.Image.open(Path(image_path)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'label': class_name,
            'image': image,
            'patient_id': patient_id,
            'x_coordinate': x_coordinate,
            'y_coordinate': y_coordinate}