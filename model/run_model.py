import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model.dataset import BreastDataset
from model.functions import extracting_coordinates, data_transformation

data_path = "/Users/bartosz/PycharmProjects/breast_cancer/data/raw_dataset/IDC_regular_ps50_idx5/"
files = os.listdir(data_path)[:20]

patient_id_list = []
patient_label_list = []
patient_label_image_path_list = []

for file in tqdm(files):
    patient_id = file
    patient_data_path = data_path + patient_id + "/"
    patient_labels = os.listdir(patient_data_path)
    for patient_label in patient_labels:
        patient_labels_path = patient_data_path + patient_label + "/"
        patient_label_images = os.listdir(patient_labels_path)
        for patient_label_image in patient_label_images:
            patient_label_image_path = patient_labels_path + patient_label_image

            patient_id_list.append(int(patient_id))
            patient_label_list.append(int(patient_label))
            patient_label_image_path_list.append(patient_label_image_path)

patients_df = pd.DataFrame(
    {'patient_id': patient_id_list,
     'label': patient_label_list,
     'image_path': patient_label_image_path_list
     })


patients_ids = patients_df.patient_id.unique()
print(patients_ids)

train_patient_ids, test_dev_patient_ids = train_test_split(patients_ids, test_size=0.4, random_state=100)
test_patient_ids, dev_patient_ids = train_test_split(test_dev_patient_ids, test_size=0.5, random_state=100)
patients_df.loc[:, "label"] = patients_df['label'].astype(np.str)

training_dataframe = patients_df[patients_df.patient_id.isin(train_patient_ids)].copy()
developing_dataframe = patients_df[patients_df.patient_id.isin(dev_patient_ids)].copy()
testing_dataframe = patients_df[patients_df.patient_id.isin(test_patient_ids)].copy()

training_dataframe = extracting_coordinates(training_dataframe)
print(training_dataframe)

developing_dataframe = extracting_coordinates(developing_dataframe)
print(developing_dataframe)

testing_dataframe = extracting_coordinates(testing_dataframe)
print(testing_dataframe)





train_dataset = BreastDataset(dataframe=training_dataframe, transform=data_transformation(data_type='training'))
dev_dataset = BreastDataset(dataframe=developing_dataframe, transform=data_transformation(data_type='validation'))
test_dataset = BreastDataset(dataframe=testing_dataframe, transform=data_transformation(data_type='validation'))


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
