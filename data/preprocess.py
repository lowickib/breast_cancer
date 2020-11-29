import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_path = "/Users/bartosz/PycharmProjects/breast_cancer/data/raw_dataset/IDC_regular_ps50_idx5/"
    files = os.listdir(data_path)[:2]

    patient_id_list = []
    patient_class_list = []
    patient_class_image_path_list = []

    for file in files:
        patient_id = file
        patient_data_path = data_path + patient_id + "/"
        patient_classes = os.listdir(patient_data_path)
        for patient_class in patient_classes:
            patient_classes_path = patient_data_path + patient_class + "/"
            patient_class_images = os.listdir(patient_classes_path)
            for patient_class_image in patient_class_images:
                patient_class_image_path = patient_classes_path + patient_class_image

                patient_id_list.append(int(patient_id))
                patient_class_list.append(int(patient_class))
                patient_class_image_path_list.append(patient_class_image_path)

    patients_df = pd.DataFrame(
        {'patient_id': patient_id_list,
         'class': patient_class_list,
         'image_path': patient_class_image_path_list
         })

    positive_cases = np.random.choice(patients_df[patients_df['class'] == 1].index.values, size=36, replace=False)
    negative_cases = np.random.choice(patients_df[patients_df['class'] == 0].index.values, size=36, replace=False)

    fig, ax = plt.subplots(6, 6, figsize=(10, 10))
    fig.suptitle('Pozytywne przypadki', y=0.92, fontsize=20)
    for x in range(6):
        for y in range(6):
            xy = positive_cases[x + y * 6]
            image = plt.imread(patients_df.loc[xy, "image_path"])
            ax[x, y].imshow(image)
            ax[x, y].grid(False)
            ax[x, y].set_xticks([])
            ax[x, y].set_yticks([])

    fig, ax = plt.subplots(6, 6, figsize=(10, 10))
    fig.suptitle('Negatywne przypadki', y=0.92, fontsize=20)
    for x in range(6):
        for y in range(6):
            xy = negative_cases[x + y * 6]
            image = plt.imread(patients_df.loc[xy, "image_path"])
            ax[x, y].imshow(image)
            ax[x, y].grid(False)
            ax[x, y].set_xticks([])
            ax[x, y].set_yticks([])

