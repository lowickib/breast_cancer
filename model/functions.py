from torchvision.transforms import transforms
import pandas as pd


def extracting_coordinates(dataframe, patient_id=None):
    if patient_id != None:
        dataframe = dataframe[dataframe['patient_id'] == patient_id]

    coordinates = dataframe['image_path'].str.split("_", n=9, expand=True)
    coordinates = coordinates.drop([0, 1, 2, 3, 4, 5, 9], axis=1)
    coordinates = coordinates.rename(columns={6: 'patient_id', 7: 'x', 8: 'y'})
    coordinates['x'] = coordinates['x'].str.replace('x', '').astype(int)
    coordinates['y'] = coordinates['y'].str.replace('y', '').astype(int)
    dataframe_coordinates_dict = {'patient_id': dataframe['patient_id'], 'x': coordinates['x'], 'y': coordinates['y'],
                                  'label': dataframe['label'], 'image_path': dataframe['image_path']}
    dataframe_coordinates = pd.DataFrame(dataframe_coordinates_dict)
    dataframe_coordinates.reset_index(drop=True, inplace=True)
    return dataframe_coordinates


def data_transformation(data_type, plot=False):
    train = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ]

    validation = [
        transforms.Resize((224, 224))
    ]

    if not plot:
        train.append(transforms.ToTensor())
        train.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        validation.append(transforms.ToTensor())
        validation.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    transformation = {
        'training': transforms.Compose(train),
        'validation': transforms.Compose(validation)
    }

    return transformation[data_type]
